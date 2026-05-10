use std::collections::BTreeMap;

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit, OsRng, Payload, rand_core::RngCore},
};
use agent_protocol::*;
use futures::future::join_all;
use provider_core::{
    ModelProvider, ProviderError, ProviderRequest, ProviderResponse, ProviderStream,
};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    #[error("unknown tool call {tool_call_id:?} in scope {scope:?}")]
    UnknownToolCall {
        scope: Box<IsolationKey>,
        tool_call_id: ToolCallId,
    },
    #[error("artifact {artifact_id:?} is not visible in scope {scope:?}")]
    MissingArtifact {
        scope: Box<IsolationKey>,
        artifact_id: ArtifactId,
    },
    #[error("artifact ref scope {artifact_scope:?} does not match request scope {request_scope:?}")]
    ArtifactScopeMismatch {
        request_scope: Box<IsolationKey>,
        artifact_scope: Box<IsolationKey>,
    },
    #[error("spawn depth {depth} for {task_id:?} exceeds max {max}")]
    SpawnDepthExceeded { task_id: TaskId, depth: u8, max: u8 },
    #[error("agent count {requested} exceeds max {max}")]
    AgentLimitExceeded { requested: u16, max: u16 },
    #[error("task {task_id:?} already exists")]
    DuplicateTask { task_id: TaskId },
    #[error("spawn budget exceeds request policy")]
    BudgetExceeded,
    #[error("provider error: {0}")]
    ProviderRejected(String),
    #[error("failed to encrypt sub-agent state")]
    EncryptionFailed,
}

impl From<ProviderError> for KernelError {
    fn from(value: ProviderError) -> Self {
        Self::ProviderRejected(value.to_string())
    }
}

#[derive(Debug, Default)]
pub struct ArtifactStore {
    artifacts: BTreeMap<(IsolationKey, ArtifactId), AgentArtifact>,
}

impl ArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, artifact: AgentArtifact) {
        let key = (artifact.scope().clone(), artifact.id());
        self.artifacts.insert(key, artifact);
    }

    pub fn get(&self, scope: &IsolationKey, artifact_id: &ArtifactId) -> Option<&AgentArtifact> {
        self.artifacts.get(&(scope.clone(), artifact_id.clone()))
    }

    pub fn contains_ref(&self, artifact_ref: &ArtifactRef) -> bool {
        self.get(&artifact_ref.scope, &artifact_ref.artifact_id)
            .is_some()
    }
}

#[derive(Debug, Default)]
pub struct ToolLedger {
    calls: BTreeMap<(IsolationKey, ToolCallId), ToolCallRecord>,
    results: BTreeMap<(IsolationKey, ToolCallId), ToolResultRecord>,
}

impl ToolLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_call(&mut self, call: ToolCallRecord) {
        let key = (call.scope.clone(), call.tool_call_id.clone());
        self.calls.insert(key, call);
    }

    pub fn record_result(&mut self, result: ToolResultRecord) -> Result<(), KernelError> {
        let key = (result.scope.clone(), result.tool_call_id.clone());
        let Some(call) = self.calls.get_mut(&key) else {
            return Err(KernelError::UnknownToolCall {
                scope: Box::new(result.scope),
                tool_call_id: result.tool_call_id,
            });
        };

        call.status = ToolCallStatus::Resolved;
        call.resolved_at_ms = Some(call.created_at_ms.saturating_add(1));
        self.results.insert(key, result);
        Ok(())
    }

    pub fn unresolved_calls(&self, scope: &IsolationKey) -> Vec<&ToolCallRecord> {
        self.calls
            .iter()
            .filter_map(|((call_scope, call_id), call)| {
                if call_scope == scope
                    && !self.results.contains_key(&(scope.clone(), call_id.clone()))
                {
                    Some(call)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn unresolved_calls_for_task(
        &self,
        scope: &IsolationKey,
        task_id: &TaskId,
    ) -> Vec<&ToolCallRecord> {
        self.unresolved_calls(scope)
            .into_iter()
            .filter(|call| &call.task_id == task_id)
            .collect()
    }

    pub fn call_count(&self) -> usize {
        self.calls.len()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelPolicy {
    pub limits: ExecutionLimits,
}

impl KernelPolicy {
    pub fn with_reasoning_effort(&self, effort: &ReasoningEffort) -> Self {
        let mut policy = self.clone();
        let max_agents = effort.max_agents();
        policy.limits.max_agents_per_request = max_agents;
        policy.limits.max_parallel_agents = max_agents;
        policy
    }
}

#[derive(Debug, Clone)]
pub struct SpawnValidator {
    policy: KernelPolicy,
}

impl SpawnValidator {
    pub fn new(policy: KernelPolicy) -> Self {
        Self { policy }
    }

    pub fn validate_and_apply(
        &self,
        scope: &IsolationKey,
        store: &ArtifactStore,
        graph: &mut TaskGraph,
        plan: &SpawnPlan,
    ) -> Result<(), KernelError> {
        let requested = graph
            .tasks
            .len()
            .saturating_add(plan.children.len())
            .try_into()
            .unwrap_or(u16::MAX);
        if requested > self.policy.limits.max_agents_per_request {
            return Err(KernelError::AgentLimitExceeded {
                requested,
                max: self.policy.limits.max_agents_per_request,
            });
        }

        if plan.budget_request.max_tokens > self.policy.limits.max_total_tokens
            || plan.budget_request.max_tool_calls > self.policy.limits.max_total_tool_calls
        {
            return Err(KernelError::BudgetExceeded);
        }

        for child in &plan.children {
            if child.spawn_depth > self.policy.limits.max_spawn_depth {
                return Err(KernelError::SpawnDepthExceeded {
                    task_id: child.task_id.clone(),
                    depth: child.spawn_depth,
                    max: self.policy.limits.max_spawn_depth,
                });
            }

            if graph.tasks.contains_key(&child.task_id) {
                return Err(KernelError::DuplicateTask {
                    task_id: child.task_id.clone(),
                });
            }

            for artifact_ref in &child.input_artifact_refs {
                if &artifact_ref.scope != scope {
                    return Err(KernelError::ArtifactScopeMismatch {
                        request_scope: Box::new(scope.clone()),
                        artifact_scope: Box::new(artifact_ref.scope.clone()),
                    });
                }

                if !store.contains_ref(artifact_ref) {
                    return Err(KernelError::MissingArtifact {
                        scope: Box::new(artifact_ref.scope.clone()),
                        artifact_id: artifact_ref.artifact_id.clone(),
                    });
                }
            }
        }

        for child in plan.children.iter().cloned() {
            graph.insert_task(child);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOutput {
    pub final_text: String,
    pub task_graph: TaskGraph,
    pub verification: VerificationReport,
    pub tool_calls: Vec<ToolCallRecord>,
    pub encrypted_subagent_state: Vec<EncryptedAgentState>,
}

#[derive(Debug, Clone)]
pub struct KernelRunner<P> {
    provider: P,
    policy: KernelPolicy,
    state_key: [u8; 32],
}

impl<P> KernelRunner<P>
where
    P: ModelProvider,
{
    pub fn new(provider: P, policy: KernelPolicy) -> Self {
        let mut state_key = [0_u8; 32];
        OsRng.fill_bytes(&mut state_key);
        Self {
            provider,
            policy,
            state_key,
        }
    }

    pub async fn run(&self, request: NormalizedRequest) -> Result<KernelOutput, KernelError> {
        let request_policy = self.policy.with_reasoning_effort(&request.reasoning_effort);
        let scope = request.isolation_key();
        let mut store = ArtifactStore::new();
        let mut ledger = ToolLedger::new();
        let mut text_artifacts = Vec::new();
        let mut encrypted_subagent_state = Vec::new();
        let system_instructions = system_instructions_with_policy(&request, &request_policy);

        for media in &request.media_artifacts {
            store.insert(AgentArtifact::Media(media.clone()));
        }

        for result in request.tool_results.clone() {
            let _ = ledger.record_result(result);
        }

        let root_task_id = TaskId::from("root");
        let root_task = SubtaskSpec {
            task_id: root_task_id.clone(),
            parent_task_id: None,
            spawn_depth: 0,
            role: AgentRole::Leader,
            objective: "produce a verified final answer".to_string(),
            input_artifact_refs: request
                .media_artifacts
                .iter()
                .map(|media| ArtifactRef {
                    scope: scope.clone(),
                    artifact_id: media.id.clone(),
                })
                .collect(),
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([
                Capability::Text,
                Capability::Image,
                Capability::Spawn,
                Capability::ToolCall,
            ]),
            limits: AgentLimits::default(),
        };

        let mut graph = TaskGraph::new(root_task_id.clone());
        graph.insert_task(root_task.clone());

        let root_response = self
            .provider
            .invoke(ProviderRequest {
                scope: scope.clone(),
                task: root_task,
                model: request.model.clone(),
                system_instructions: system_instructions.clone(),
                thinking_enabled: request.thinking_enabled,
                thinking_format: request.thinking_format.clone(),
                input_text: flatten_text(&request),
                messages: request.messages.clone(),
                media_artifacts: request.media_artifacts.clone(),
                artifacts: Vec::new(),
                tools: request.tools.clone(),
                tool_choice: request.tool_choice.clone(),
                parallel_tool_calls: request.parallel_tool_calls,
                tool_results: request.tool_results.clone(),
            })
            .await?;

        Self::apply_provider_response(
            &request_policy,
            &scope,
            &mut store,
            &mut ledger,
            &mut graph,
            &mut text_artifacts,
            root_response,
        )?;

        let tasks: Vec<SubtaskSpec> = graph
            .tasks
            .values()
            .filter(|task| task.parent_task_id.is_some())
            .cloned()
            .collect();
        let has_child_tasks = !tasks.is_empty();

        let child_futures = tasks.into_iter().map(|task| {
            let artifacts = task
                .input_artifact_refs
                .iter()
                .filter_map(|artifact_ref| {
                    store
                        .get(&artifact_ref.scope, &artifact_ref.artifact_id)
                        .cloned()
                })
                .collect();
            let request = ProviderRequest {
                scope: scope.clone(),
                task: task.clone(),
                model: request.model.clone(),
                system_instructions: system_instructions.clone(),
                thinking_enabled: request.thinking_enabled,
                thinking_format: request.thinking_format.clone(),
                input_text: task.objective.clone(),
                messages: Vec::new(),
                media_artifacts: request.media_artifacts.clone(),
                artifacts,
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                parallel_tool_calls: None,
                tool_results: Vec::new(),
            };
            async move {
                let response = self.provider.invoke(request).await?;
                Ok::<_, ProviderError>((task, response))
            }
        });

        let child_results = join_all(child_futures).await;
        for child_result in child_results {
            let (task, child_response) = child_result?;
            encrypted_subagent_state.push(self.seal_subagent_state(
                &scope,
                &task,
                &child_response,
            )?);

            Self::apply_provider_response(
                &request_policy,
                &scope,
                &mut store,
                &mut ledger,
                &mut graph,
                &mut text_artifacts,
                child_response,
            )?;
        }

        let root_unresolved: Vec<ToolCallId> = ledger
            .unresolved_calls_for_task(&scope, &root_task_id)
            .into_iter()
            .map(|call| call.tool_call_id.clone())
            .collect();

        let public_tool_calls: Vec<ToolCallRecord> = ledger
            .unresolved_calls_for_task(&scope, &root_task_id)
            .into_iter()
            .cloned()
            .collect();

        let final_text = if root_unresolved.is_empty() && has_child_tasks {
            let synth_task = SubtaskSpec {
                task_id: TaskId::from("synthesizer"),
                parent_task_id: Some(root_task_id.clone()),
                spawn_depth: 0,
                role: AgentRole::Synthesizer,
                objective: "synthesize a natural final answer for the user".to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits::default(),
            };
            let synth_response = self
                .provider
                .invoke(ProviderRequest {
                    scope: scope.clone(),
                    task: synth_task,
                    model: request.model.clone(),
                    system_instructions: system_instructions.clone(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: synthesis_input(&request, &text_artifacts),
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: text_artifacts.clone(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                })
                .await?;
            synthesize_text(&synth_response.artifacts)
        } else if root_unresolved.is_empty() {
            synthesize_text(&text_artifacts)
        } else {
            String::new()
        };

        let verification = VerificationReport {
            request_id: request.request_id.clone(),
            passed: root_unresolved.is_empty(),
            issues: if root_unresolved.is_empty() {
                Vec::new()
            } else {
                vec![VerificationIssue {
                    code: "unresolved_tool_calls".to_string(),
                    message: "tool calls must be resolved before final synthesis".to_string(),
                }]
            },
            artifact_coverage: text_artifacts
                .iter()
                .map(|artifact| ArtifactCoverage {
                    artifact_id: artifact.id(),
                    kind: ArtifactKind::Text,
                    covered: true,
                })
                .collect(),
            unresolved_tool_calls: root_unresolved,
            budget_summary: BudgetSummary {
                token_budget: request_policy.limits.max_total_tokens,
                token_used: 0,
                tool_call_budget: request_policy.limits.max_total_tool_calls,
                tool_calls_used: ledger.call_count().try_into().unwrap_or(u16::MAX),
            },
        };

        Ok(KernelOutput {
            final_text,
            task_graph: graph,
            verification,
            tool_calls: public_tool_calls,
            encrypted_subagent_state,
        })
    }

    pub async fn stream_root(
        &self,
        request: NormalizedRequest,
    ) -> Result<ProviderStream, KernelError> {
        let request_policy = self.policy.with_reasoning_effort(&request.reasoning_effort);
        let scope = request.isolation_key();
        let root_task = root_task(&scope, &request);
        self.provider
            .stream(ProviderRequest {
                scope,
                task: root_task,
                model: request.model.clone(),
                system_instructions: system_instructions_with_policy(&request, &request_policy),
                thinking_enabled: request.thinking_enabled,
                thinking_format: request.thinking_format.clone(),
                input_text: flatten_text(&request),
                messages: request.messages.clone(),
                media_artifacts: request.media_artifacts.clone(),
                artifacts: Vec::new(),
                tools: request.tools.clone(),
                tool_choice: request.tool_choice.clone(),
                parallel_tool_calls: request.parallel_tool_calls,
                tool_results: request.tool_results.clone(),
            })
            .await
            .map_err(KernelError::from)
    }

    fn apply_provider_response(
        policy: &KernelPolicy,
        scope: &IsolationKey,
        store: &mut ArtifactStore,
        ledger: &mut ToolLedger,
        graph: &mut TaskGraph,
        text_artifacts: &mut Vec<AgentArtifact>,
        response: ProviderResponse,
    ) -> Result<(), KernelError> {
        if ledger
            .call_count()
            .saturating_add(response.tool_calls.len())
            > policy.limits.max_total_tool_calls as usize
        {
            return Err(KernelError::BudgetExceeded);
        }

        for call in response.tool_calls {
            ledger.record_call(call);
        }

        for artifact in response.artifacts {
            match artifact {
                AgentArtifact::SpawnPlan { plan, .. } => {
                    SpawnValidator::new(policy.clone())
                        .validate_and_apply(scope, store, graph, &plan)?;
                }
                AgentArtifact::Text { .. } => {
                    store.insert(artifact.clone());
                    text_artifacts.push(artifact);
                }
                _ => {
                    store.insert(artifact);
                }
            }
        }

        Ok(())
    }

    fn seal_subagent_state(
        &self,
        scope: &IsolationKey,
        task: &SubtaskSpec,
        response: &ProviderResponse,
    ) -> Result<EncryptedAgentState, KernelError> {
        let plaintext = json!({
            "scope": scope,
            "task": task,
            "response": response,
        })
        .to_string();
        let aad = format!(
            "{}:{}:{}",
            scope.request_id.as_ref(),
            scope.conversation_fingerprint.as_ref(),
            task.task_id.as_ref()
        );
        let cipher = Aes256Gcm::new_from_slice(&self.state_key)
            .map_err(|_| KernelError::EncryptionFailed)?;
        let mut nonce_bytes = [0_u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce_bytes),
                Payload {
                    msg: plaintext.as_bytes(),
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| KernelError::EncryptionFailed)?;

        Ok(EncryptedAgentState {
            task_id: task.task_id.clone(),
            algorithm: "AES-256-GCM".to_string(),
            nonce: hex::encode(nonce_bytes),
            ciphertext: hex::encode(ciphertext),
            aad,
        })
    }
}

fn root_task(scope: &IsolationKey, request: &NormalizedRequest) -> SubtaskSpec {
    SubtaskSpec {
        task_id: TaskId::from("root"),
        parent_task_id: None,
        spawn_depth: 0,
        role: AgentRole::Leader,
        objective: "produce a verified final answer".to_string(),
        input_artifact_refs: request
            .media_artifacts
            .iter()
            .map(|media| ArtifactRef {
                scope: scope.clone(),
                artifact_id: media.id.clone(),
            })
            .collect(),
        expected_outputs: vec![ArtifactKind::Text],
        allowed_capabilities: CapabilitySet::from([
            Capability::Text,
            Capability::Image,
            Capability::Spawn,
            Capability::ToolCall,
        ]),
        limits: AgentLimits::default(),
    }
}

#[derive(Debug, Clone, Default)]
pub struct MockProvider;

#[async_trait::async_trait]
impl ModelProvider for MockProvider {
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        if request.task.role == AgentRole::Leader && !request.tool_results.is_empty() {
            let results = request
                .tool_results
                .iter()
                .map(|result| format!("{}={}", result.tool_call_id.as_ref(), result.result_json))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: format!("Here is a clear, usable answer from tool results: {results}"),
                }],
                tool_calls: Vec::new(),
                usage: Default::default(),
            });
        }

        if request.task.role == AgentRole::Leader && request.input_text.contains("tool") {
            return Ok(ProviderResponse {
                artifacts: Vec::new(),
                tool_calls: vec![ToolCallRecord {
                    tool_call_id: ToolCallId::from("call-1"),
                    scope: request.scope,
                    task_id: request.task.task_id,
                    agent_id: AgentId::from("agent-root"),
                    tool_name: "lookup".to_string(),
                    arguments_json: serde_json::json!({"query": "required"}),
                    arguments_sha256: "mock-args".to_string(),
                    status: ToolCallStatus::Pending,
                    created_at_ms: 1,
                    resolved_at_ms: None,
                }],
                usage: Default::default(),
            });
        }

        if request.task.role == AgentRole::Leader && request.input_text.contains("spawn") {
            let image_ref = request
                .media_artifacts
                .first()
                .map(|media| ArtifactRef {
                    scope: request.scope.clone(),
                    artifact_id: media.id.clone(),
                })
                .into_iter()
                .collect();
            let child = SubtaskSpec {
                task_id: TaskId::from("child-1"),
                parent_task_id: Some(request.task.task_id.clone()),
                spawn_depth: request.task.spawn_depth.saturating_add(1),
                role: AgentRole::Worker,
                objective: "child visual inspection".to_string(),
                input_artifact_refs: image_ref,
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text, Capability::Image]),
                limits: AgentLimits::default(),
            };
            return Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::SpawnPlan {
                    id: ArtifactId::from("spawn-plan-1"),
                    scope: request.scope.clone(),
                    plan: SpawnPlan {
                        parent_task_id: request.task.task_id,
                        reason: "Need bounded child inspection".to_string(),
                        children: vec![child],
                        expected_artifacts: vec![ArtifactKind::Text],
                        budget_request: BudgetRequest {
                            max_tokens: 256,
                            max_tool_calls: 0,
                        },
                    },
                }],
                tool_calls: Vec::new(),
                usage: Default::default(),
            });
        }

        let text = match request.task.role {
            AgentRole::Worker => format!("child completed: {}", request.task.objective),
            AgentRole::Synthesizer => {
                "Here is a clear, usable answer based on the verified agent results.".to_string()
            }
            _ => format!("Here is a clear, usable answer: {}", request.input_text),
        };

        Ok(ProviderResponse {
            artifacts: vec![AgentArtifact::Text {
                id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                scope: request.scope,
                text,
            }],
            tool_calls: Vec::new(),
            usage: Default::default(),
        })
    }
}

fn flatten_text(request: &NormalizedRequest) -> String {
    request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(text.as_str()),
            NormalizedContentPart::ToolCall { .. } => None,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn system_instructions(request: &NormalizedRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|message| message.role == MessageRole::System)
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn system_instructions_with_policy(
    request: &NormalizedRequest,
    policy: &KernelPolicy,
) -> Vec<String> {
    let mut instructions = system_instructions(request);
    instructions.push(orchestration_policy_instruction(
        &request.reasoning_effort,
        policy,
    ));
    instructions
}

fn orchestration_policy_instruction(effort: &ReasoningEffort, policy: &KernelPolicy) -> String {
    let max_agents = policy.limits.max_agents_per_request;
    let target_parallel_agents = max_agents.saturating_sub(1);
    format!(
        "Orchestration policy: reasoning_effort={}; max_agents_per_request={max_agents}; max_parallel_agents={}; target_parallel_agents={target_parallel_agents}; max_spawn_depth={}; max_total_tool_calls={}; max_total_tokens={}. For complex or ambiguous tasks, use deterministic decomposition and spawn up to target_parallel_agents independent child agents when that increases coverage. Low/medium should stay compact; high/xhigh should broaden coverage and verification. Only root-visible tool calls and the final synthesis are public.",
        reasoning_effort_name(effort),
        policy.limits.max_parallel_agents,
        policy.limits.max_spawn_depth,
        policy.limits.max_total_tool_calls,
        policy.limits.max_total_tokens,
    )
}

fn reasoning_effort_name(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    }
}

fn synthesis_input(request: &NormalizedRequest, artifacts: &[AgentArtifact]) -> String {
    let original = flatten_text(request);
    let artifact_text = synthesize_text(artifacts);
    format!(
        "User request:\n{original}\n\nVerified internal findings:\n{artifact_text}\n\nReturn only a natural final answer. Do not expose sub-agent state, internal tool calls, raw artifacts, or orchestration details."
    )
}

fn synthesize_text(artifacts: &[AgentArtifact]) -> String {
    artifacts
        .iter()
        .filter_map(|artifact| match artifact {
            AgentArtifact::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn crate_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn same_id_cross_scope_artifact_lookup_is_rejected() {
        let scope_a = IsolationKey::new("tenant", "request-a", "conversation");
        let scope_b = IsolationKey::new("tenant", "request-b", "conversation");
        let artifact_id = ArtifactId::from("shared-id");
        let mut store = ArtifactStore::new();

        store.insert(AgentArtifact::Text {
            id: artifact_id.clone(),
            scope: scope_a.clone(),
            text: "request a".to_string(),
        });
        store.insert(AgentArtifact::Text {
            id: artifact_id.clone(),
            scope: scope_b.clone(),
            text: "request b".to_string(),
        });

        let found_a = store.get(&scope_a, &artifact_id).unwrap();
        let found_b = store.get(&scope_b, &artifact_id).unwrap();
        assert_ne!(found_a, found_b);
        assert!(
            store
                .get(
                    &IsolationKey::new("tenant", "request-c", "conversation"),
                    &artifact_id
                )
                .is_none()
        );
    }

    #[test]
    fn tool_ledger_matches_results_by_scope_and_call_id() {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        let other_scope = IsolationKey::new("tenant", "request-b", "conversation");
        let mut ledger = ToolLedger::new();
        let call = ToolCallRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope: scope.clone(),
            task_id: TaskId::from("task-root"),
            agent_id: AgentId::from("agent-root"),
            tool_name: "lookup".to_string(),
            arguments_json: serde_json::json!({"q": "rust"}),
            arguments_sha256: "hash".to_string(),
            status: ToolCallStatus::Pending,
            created_at_ms: 1,
            resolved_at_ms: None,
        };
        ledger.record_call(call);

        let wrong_scope_result = ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope: other_scope,
            result_json: serde_json::json!({"ok": true}),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        };
        assert!(matches!(
            ledger.record_result(wrong_scope_result),
            Err(KernelError::UnknownToolCall { .. })
        ));

        let result = ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope: scope.clone(),
            result_json: serde_json::json!({"ok": true}),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        };
        ledger.record_result(result).unwrap();
        assert_eq!(ledger.unresolved_calls(&scope).len(), 0);
    }

    #[test]
    fn spawn_validator_rejects_depth_and_agent_count() {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        let store = ArtifactStore::new();
        let root = TaskId::from("root");
        let mut graph = TaskGraph::new(root.clone());
        let validator = SpawnValidator::new(KernelPolicy {
            limits: ExecutionLimits {
                max_spawn_depth: 1,
                max_agents_per_request: 1,
                ..ExecutionLimits::default()
            },
        });

        let too_deep = SpawnPlan {
            parent_task_id: root.clone(),
            reason: "too deep".to_string(),
            children: vec![child("child-deep", &root, 2, &scope)],
            expected_artifacts: vec![ArtifactKind::Text],
            budget_request: BudgetRequest {
                max_tokens: 10,
                max_tool_calls: 0,
            },
        };
        assert!(matches!(
            validator.validate_and_apply(&scope, &store, &mut graph, &too_deep),
            Err(KernelError::SpawnDepthExceeded { .. })
        ));

        let too_many = SpawnPlan {
            parent_task_id: root.clone(),
            reason: "too many".to_string(),
            children: vec![
                child("child-a", &root, 1, &scope),
                child("child-b", &root, 1, &scope),
            ],
            expected_artifacts: vec![ArtifactKind::Text],
            budget_request: BudgetRequest {
                max_tokens: 10,
                max_tool_calls: 0,
            },
        };
        assert!(matches!(
            validator.validate_and_apply(&scope, &store, &mut graph, &too_many),
            Err(KernelError::AgentLimitExceeded { .. })
        ));
    }

    #[test]
    fn valid_spawn_plan_expands_graph() {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        let root = TaskId::from("root");
        let mut store = ArtifactStore::new();
        let image_id = ArtifactId::from("image-1");
        store.insert(AgentArtifact::Media(MediaArtifact {
            id: image_id.clone(),
            scope: scope.clone(),
            media_type: "image/png".to_string(),
            source: MediaSource::Base64 {
                data: "AAAA".to_string(),
            },
            sha256: "hash".to_string(),
            byte_len: Some(4),
        }));

        let mut graph = TaskGraph::new(root.clone());
        let validator = SpawnValidator::new(KernelPolicy::default());
        let plan = SpawnPlan {
            parent_task_id: root.clone(),
            reason: "inspect image".to_string(),
            children: vec![SubtaskSpec {
                input_artifact_refs: vec![ArtifactRef {
                    scope: scope.clone(),
                    artifact_id: image_id,
                }],
                ..child("child-image", &root, 1, &scope)
            }],
            expected_artifacts: vec![ArtifactKind::Text],
            budget_request: BudgetRequest {
                max_tokens: 64,
                max_tool_calls: 0,
            },
        };

        validator
            .validate_and_apply(&scope, &store, &mut graph, &plan)
            .unwrap();
        assert!(graph.tasks.contains_key(&TaskId::from("child-image")));
    }

    #[tokio::test]
    async fn runs_spawn_plan_through_bounded_kernel() {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        let image = MediaArtifact {
            id: ArtifactId::from("image-1"),
            scope: scope.clone(),
            media_type: "image/png".to_string(),
            source: MediaSource::Base64 {
                data: "AAAA".to_string(),
            },
            sha256: "hash".to_string(),
            byte_len: Some(4),
        };
        let request = NormalizedRequest {
            request_id: scope.request_id.clone(),
            tenant_id: scope.tenant_id.clone(),
            conversation_fingerprint: scope.conversation_fingerprint.clone(),
            source_format: SourceFormat::OpenAIChat,
            model: "mock".to_string(),
            messages: vec![NormalizedMessage {
                role: MessageRole::User,
                content: vec![NormalizedContentPart::Text {
                    text: "spawn visual inspection".to_string(),
                }],
            }],
            media_artifacts: vec![image],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            tool_results: vec![],
            stream: false,
            thinking_enabled: false,
            thinking_format: ThinkingFormat::Auto,
            reasoning_effort: ReasoningEffort::Medium,
            metadata: serde_json::json!({}),
        };

        let runner = KernelRunner::new(MockProvider, KernelPolicy::default());
        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert!(output.final_text.contains("clear, usable answer"));
        assert!(!output.final_text.contains("child completed"));
        assert_eq!(output.encrypted_subagent_state.len(), 1);
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("child-1"))
        );
    }

    #[tokio::test]
    async fn only_root_tool_calls_are_public_and_child_state_is_encrypted() {
        let request = text_request("spawn child tool");
        let runner = KernelRunner::new(ChildToolProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.tool_calls.is_empty());
        assert_eq!(output.final_text, "Final answer from main synthesizer.");
        assert_eq!(output.encrypted_subagent_state.len(), 1);
        let sealed = &output.encrypted_subagent_state[0];
        assert_eq!(sealed.task_id.as_ref(), "child-tool");
        assert!(!sealed.ciphertext.contains("child-secret-output"));
        assert!(!sealed.ciphertext.contains("private_lookup"));
    }

    #[tokio::test]
    async fn child_tasks_run_in_parallel() {
        let mut request = text_request("spawn slow children");
        request.reasoning_effort = ReasoningEffort::High;
        let runner = KernelRunner::new(SlowFanoutProvider, KernelPolicy::default());

        let started = Instant::now();
        let output = runner.run(request).await.unwrap();
        let elapsed = started.elapsed();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 4);
        assert!(
            elapsed < Duration::from_millis(260),
            "children appear sequential; elapsed={elapsed:?}"
        );
    }

    #[tokio::test]
    async fn root_only_request_does_not_call_synthesizer() {
        let request = text_request("simple answer");
        let provider = CountingProvider::default();
        let calls = provider.calls.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(output.final_text, "root final answer");
    }

    #[tokio::test]
    async fn tool_call_budget_summary_counts_provider_calls() {
        let request = text_request("please use a tool");
        let runner = KernelRunner::new(MockProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(!output.verification.passed);
        assert_eq!(output.verification.budget_summary.tool_calls_used, 1);
        assert_eq!(output.verification.budget_summary.tool_call_budget, 16);
    }

    #[tokio::test]
    async fn client_tools_are_available_only_to_root_agent() {
        let mut request = text_request("spawn tool visibility");
        request.tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: Some("public client tool".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let provider = ToolVisibilityProvider::default();
        let seen = provider.seen.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen, vec!["Leader:1", "Worker:0", "Synthesizer:0"]);
    }

    #[tokio::test]
    async fn reasoning_effort_medium_limits_request_to_four_agents() {
        let mut request = text_request("spawn five children");
        request.reasoning_effort = ReasoningEffort::Medium;
        let runner = KernelRunner::new(FiveChildProvider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(
            error,
            KernelError::AgentLimitExceeded {
                requested: 6,
                max: 4
            }
        ));
    }

    #[tokio::test]
    async fn provider_tool_calls_cannot_exceed_total_budget() {
        let mut request = text_request("too many tools");
        request.reasoning_effort = ReasoningEffort::Low;
        let runner = KernelRunner::new(ManyToolCallsProvider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(error, KernelError::BudgetExceeded));
    }

    #[tokio::test]
    async fn reasoning_effort_high_allows_more_than_four_agents() {
        let mut request = text_request("spawn five children");
        request.reasoning_effort = ReasoningEffort::High;
        let runner = KernelRunner::new(FiveChildProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 5);
    }

    #[tokio::test]
    async fn root_agent_receives_reasoning_effort_execution_profile() {
        let mut request = text_request("complex market, technical, legal, and risk analysis");
        request.reasoning_effort = ReasoningEffort::High;
        let provider = EffortInstructionProbe::default();
        let seen = provider.seen.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        let instructions = seen.lock().unwrap().join("\n");
        assert!(instructions.contains("reasoning_effort=high"));
        assert!(instructions.contains("max_agents_per_request=16"));
        assert!(instructions.contains("target_parallel_agents=15"));
    }

    #[tokio::test]
    async fn higher_reasoning_effort_improves_deterministic_eval_coverage() {
        let low = run_effort_coverage_eval(ReasoningEffort::Low).await;
        let high = run_effort_coverage_eval(ReasoningEffort::High).await;
        let xhigh = run_effort_coverage_eval(ReasoningEffort::XHigh).await;

        assert_eq!(low.coverage_score, 3);
        assert_eq!(high.coverage_score, 8);
        assert_eq!(xhigh.coverage_score, 8);
        assert!(high.coverage_score > low.coverage_score);
        assert!(xhigh.coverage_score >= high.coverage_score);
        assert_eq!(low.child_agents, 3);
        assert_eq!(high.child_agents, 8);
    }

    struct EffortCoverageEval {
        coverage_score: usize,
        child_agents: usize,
    }

    async fn run_effort_coverage_eval(effort: ReasoningEffort) -> EffortCoverageEval {
        let mut request = text_request("deep compare across eight independent dimensions");
        request.reasoning_effort = effort;
        let runner = KernelRunner::new(EffortCoverageProvider, KernelPolicy::default());
        let output = runner.run(request).await.unwrap();
        let coverage_score = output
            .final_text
            .strip_prefix("coverage_score=")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap();

        EffortCoverageEval {
            coverage_score,
            child_agents: output.encrypted_subagent_state.len(),
        }
    }

    fn text_request(text: &str) -> NormalizedRequest {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        NormalizedRequest {
            request_id: scope.request_id.clone(),
            tenant_id: scope.tenant_id.clone(),
            conversation_fingerprint: scope.conversation_fingerprint.clone(),
            source_format: SourceFormat::OpenAIChat,
            model: "mock".to_string(),
            messages: vec![NormalizedMessage {
                role: MessageRole::User,
                content: vec![NormalizedContentPart::Text {
                    text: text.to_string(),
                }],
            }],
            media_artifacts: vec![],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            tool_results: vec![],
            stream: false,
            thinking_enabled: false,
            thinking_format: ThinkingFormat::Auto,
            reasoning_effort: ReasoningEffort::Medium,
            metadata: serde_json::json!({}),
        }
    }

    #[derive(Clone)]
    struct ChildToolProvider;

    #[async_trait::async_trait]
    impl ModelProvider for ChildToolProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-child-tool"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: request.task.task_id,
                            reason: "delegate private lookup".to_string(),
                            children: vec![SubtaskSpec {
                                task_id: TaskId::from("child-tool"),
                                parent_task_id: Some(TaskId::from("root")),
                                spawn_depth: 1,
                                role: AgentRole::Worker,
                                objective: "private child tool work".to_string(),
                                input_artifact_refs: vec![],
                                expected_outputs: vec![ArtifactKind::Text],
                                allowed_capabilities: CapabilitySet::from([
                                    Capability::Text,
                                    Capability::ToolCall,
                                ]),
                                limits: AgentLimits::default(),
                            }],
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 128,
                                max_tool_calls: 1,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Worker => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("child-secret-text"),
                        scope: request.scope.clone(),
                        text: "child-secret-output".to_string(),
                    }],
                    tool_calls: vec![ToolCallRecord {
                        tool_call_id: ToolCallId::from("child-call"),
                        scope: request.scope,
                        task_id: request.task.task_id,
                        agent_id: AgentId::from("child-agent"),
                        tool_name: "private_lookup".to_string(),
                        arguments_json: serde_json::json!({"secret": true}),
                        arguments_sha256: "hash".to_string(),
                        status: ToolCallStatus::Pending,
                        created_at_ms: 1,
                        resolved_at_ms: None,
                    }],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("final"),
                        scope: request.scope,
                        text: "Final answer from main synthesizer.".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                _ => Ok(ProviderResponse::default()),
            }
        }
    }

    #[derive(Clone, Default)]
    struct CountingProvider {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for CountingProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let text = match request.task.role {
                AgentRole::Leader => "root final answer",
                AgentRole::Synthesizer => "unexpected synth",
                _ => "unexpected worker",
            };
            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: text.to_string(),
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone, Default)]
    struct ToolVisibilityProvider {
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for ToolVisibilityProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.seen.lock().unwrap().push(format!(
                "{:?}:{}",
                request.task.role,
                request.tools.len()
            ));
            match request.task.role {
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-tool-visibility"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: request.task.task_id,
                            reason: "verify tool visibility".to_string(),
                            children: vec![SubtaskSpec {
                                task_id: TaskId::from("tool-visibility-child"),
                                parent_task_id: Some(TaskId::from("root")),
                                spawn_depth: 1,
                                role: AgentRole::Worker,
                                objective: "child should not see client tools".to_string(),
                                input_artifact_refs: vec![],
                                expected_outputs: vec![ArtifactKind::Text],
                                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                                limits: AgentLimits::default(),
                            }],
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 128,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Worker | AgentRole::Synthesizer | AgentRole::Verifier => {
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                            scope: request.scope,
                            text: "done".to_string(),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
            }
        }
    }

    #[derive(Clone)]
    struct SlowFanoutProvider;

    #[async_trait::async_trait]
    impl ModelProvider for SlowFanoutProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-slow"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: request.task.task_id,
                            reason: "parallel fanout".to_string(),
                            children: (0..4)
                                .map(|index| SubtaskSpec {
                                    task_id: TaskId::from(format!("slow-child-{index}")),
                                    parent_task_id: Some(TaskId::from("root")),
                                    spawn_depth: 1,
                                    role: AgentRole::Worker,
                                    objective: format!("slow work {index}"),
                                    input_artifact_refs: vec![],
                                    expected_outputs: vec![ArtifactKind::Text],
                                    allowed_capabilities: CapabilitySet::from([Capability::Text]),
                                    limits: AgentLimits::default(),
                                })
                                .collect(),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Worker => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                            scope: request.scope,
                            text: request.task.objective,
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("final"),
                        scope: request.scope,
                        text: "Parallel synthesis complete.".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                _ => Ok(ProviderResponse::default()),
            }
        }
    }

    #[derive(Clone)]
    struct FiveChildProvider;

    #[async_trait::async_trait]
    impl ModelProvider for FiveChildProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-five"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: request.task.task_id,
                            reason: "five child fanout".to_string(),
                            children: (0..5)
                                .map(|index| SubtaskSpec {
                                    task_id: TaskId::from(format!("child-{index}")),
                                    parent_task_id: Some(TaskId::from("root")),
                                    spawn_depth: 1,
                                    role: AgentRole::Worker,
                                    objective: format!("work {index}"),
                                    input_artifact_refs: vec![],
                                    expected_outputs: vec![ArtifactKind::Text],
                                    allowed_capabilities: CapabilitySet::from([Capability::Text]),
                                    limits: AgentLimits::default(),
                                })
                                .collect(),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Worker | AgentRole::Synthesizer | AgentRole::Verifier => {
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                            scope: request.scope,
                            text: request.task.objective,
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
            }
        }
    }

    #[derive(Clone, Default)]
    struct EffortInstructionProbe {
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for EffortInstructionProbe {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.seen
                .lock()
                .unwrap()
                .extend(request.system_instructions.clone());
            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from("probe-text"),
                    scope: request.scope,
                    text: "profile observed".to_string(),
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone)]
    struct EffortCoverageProvider;

    #[async_trait::async_trait]
    impl ModelProvider for EffortCoverageProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => {
                    let target = target_parallel_agents(&request.system_instructions).min(8);
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-effort-coverage"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: request.task.task_id,
                                reason: "cover independent evaluation dimensions".to_string(),
                                children: (0..target)
                                    .map(|index| SubtaskSpec {
                                        task_id: TaskId::from(format!("coverage-child-{index}")),
                                        parent_task_id: Some(TaskId::from("root")),
                                        spawn_depth: 1,
                                        role: AgentRole::Worker,
                                        objective: format!("cover dimension {index}"),
                                        input_artifact_refs: vec![],
                                        expected_outputs: vec![ArtifactKind::Text],
                                        allowed_capabilities: CapabilitySet::from([
                                            Capability::Text,
                                        ]),
                                        limits: AgentLimits::default(),
                                    })
                                    .collect(),
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 1024,
                                    max_tool_calls: 0,
                                },
                            },
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Worker => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: format!("covered dimension: {}", request.task.objective),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => {
                    let score = request.input_text.matches("covered dimension:").count();
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("coverage-score"),
                            scope: request.scope,
                            text: format!("coverage_score={score}"),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Verifier => Ok(ProviderResponse::default()),
            }
        }
    }

    fn target_parallel_agents(instructions: &[String]) -> usize {
        instructions
            .iter()
            .flat_map(|instruction| instruction.split([';', '\n']))
            .filter_map(|part| part.trim().strip_prefix("target_parallel_agents="))
            .filter_map(|value| value.parse::<usize>().ok())
            .next()
            .unwrap_or(1)
    }

    #[derive(Clone)]
    struct ManyToolCallsProvider;

    #[async_trait::async_trait]
    impl ModelProvider for ManyToolCallsProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse {
                artifacts: Vec::new(),
                tool_calls: (0..17)
                    .map(|index| ToolCallRecord {
                        tool_call_id: ToolCallId::from(format!("call-{index}")),
                        scope: request.scope.clone(),
                        task_id: request.task.task_id.clone(),
                        agent_id: AgentId::from("agent-root"),
                        tool_name: "lookup".to_string(),
                        arguments_json: serde_json::json!({"index": index}),
                        arguments_sha256: format!("hash-{index}"),
                        status: ToolCallStatus::Pending,
                        created_at_ms: index,
                        resolved_at_ms: None,
                    })
                    .collect(),
                usage: Default::default(),
            })
        }
    }

    fn child(id: &str, parent: &TaskId, depth: u8, scope: &IsolationKey) -> SubtaskSpec {
        SubtaskSpec {
            task_id: TaskId::from(id),
            parent_task_id: Some(parent.clone()),
            spawn_depth: depth,
            role: AgentRole::Worker,
            objective: "work".to_string(),
            input_artifact_refs: vec![ArtifactRef {
                scope: scope.clone(),
                artifact_id: ArtifactId::from("input"),
            }],
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([Capability::Text]),
            limits: AgentLimits::default(),
        }
    }
}
