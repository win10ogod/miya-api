use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit, OsRng, Payload, rand_core::RngCore},
};
use agent_protocol::*;
use futures::{StreamExt, stream};
use provider_core::{
    ModelProvider, ProviderError, ProviderRequest, ProviderResponse, ProviderStream, ProviderUsage,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

const MAX_ORCHESTRATION_REPAIR_ATTEMPTS: usize = 3;
const DEFAULT_MAX_SEMANTIC_REPAIR_ATTEMPTS: u8 = 2;

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
    #[error("spawn tool-call budget exceeds request policy")]
    BudgetExceeded,
    #[error("provider error: {0}")]
    Provider(ProviderError),
    #[error("provider error: {0}")]
    ProviderRejected(String),
    #[error("request execution exceeded {timeout_ms} ms")]
    RequestTimeout { timeout_ms: u64 },
    #[error("agent {task_id:?} execution exceeded {timeout_ms} ms")]
    AgentTimeout { task_id: TaskId, timeout_ms: u64 },
    #[error("failed to encrypt sub-agent state")]
    EncryptionFailed,
}

impl From<ProviderError> for KernelError {
    fn from(value: ProviderError) -> Self {
        Self::Provider(value)
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
    pub semantic_verification: SemanticVerificationPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticVerificationPolicy {
    pub enabled: bool,
    pub max_repair_attempts: u8,
}

impl Default for SemanticVerificationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_repair_attempts: DEFAULT_MAX_SEMANTIC_REPAIR_ATTEMPTS,
        }
    }
}

impl KernelPolicy {
    pub fn with_reasoning_effort(&self, effort: &ReasoningEffort) -> Self {
        let mut policy = self.clone();
        let max_agents = effort.max_agents();
        policy.limits.max_agents_per_request = max_agents;
        policy.limits.max_parallel_agents = bounded_parallel_agents(
            policy.limits.max_parallel_agents,
            policy.limits.max_agents_per_request,
        );
        policy
    }

    pub fn for_request(&self, request: &NormalizedRequest) -> Self {
        self.with_reasoning_effort(&request.reasoning_effort)
            .with_metadata_overrides(&request.metadata)
    }

    fn with_metadata_overrides(mut self, metadata: &serde_json::Value) -> Self {
        if let Some(max_parallel_agents) = metadata_u16(
            metadata,
            &[
                &["max_parallel_agents"],
                &["parallel_agents"],
                &["agent", "max_parallel_agents"],
                &["agent", "parallel_agents"],
                &["agent", "parallelism"],
                &["orchestration", "max_parallel_agents"],
                &["orchestration", "parallel_agents"],
            ],
        ) {
            self.limits.max_parallel_agents =
                bounded_parallel_agents(max_parallel_agents, self.limits.max_agents_per_request);
        }
        self
    }
}

fn bounded_parallel_agents(requested: u16, max_agents_per_request: u16) -> u16 {
    if max_agents_per_request == 0 {
        0
    } else {
        requested.clamp(1, max_agents_per_request)
    }
}

fn metadata_u16(metadata: &serde_json::Value, paths: &[&[&str]]) -> Option<u16> {
    paths.iter().find_map(|path| {
        let mut value = metadata;
        for segment in *path {
            value = value.get(*segment)?;
        }
        match value {
            serde_json::Value::Number(number) => number.as_u64(),
            serde_json::Value::String(text) => text.trim().parse::<u64>().ok(),
            _ => None,
        }
        .and_then(|value| value.try_into().ok())
    })
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
            .values()
            .filter(|task| task.parent_task_id.is_some())
            .count()
            .saturating_add(plan.children.len())
            .try_into()
            .unwrap_or(u16::MAX);
        if requested > self.policy.limits.max_agents_per_request {
            return Err(KernelError::AgentLimitExceeded {
                requested,
                max: self.policy.limits.max_agents_per_request,
            });
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

#[derive(Debug, Deserialize)]
struct SemanticVerdict {
    passed: bool,
    #[serde(default)]
    issues: Vec<VerificationIssue>,
    #[serde(default)]
    covered_artifact_ids: Vec<String>,
}

struct SemanticVerificationResult {
    final_text: String,
    passed: bool,
    issues: Vec<VerificationIssue>,
    covered_artifact_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOutput {
    pub final_text: String,
    pub task_graph: TaskGraph,
    pub verification: VerificationReport,
    pub tool_calls: Vec<ToolCallRecord>,
    pub encrypted_subagent_state: Vec<EncryptedAgentState>,
    pub usage: ProviderUsage,
    pub provider_call_count: u32,
    pub trace_events: Vec<KernelTraceEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelTraceEvent {
    AgentInput {
        task_id: TaskId,
        role: AgentRole,
        objective: String,
        input_text: String,
    },
    SpawnPlan {
        task_id: TaskId,
        reason: String,
        children: Vec<SubtaskSpec>,
    },
    AgentOutput {
        task_id: TaskId,
        role: AgentRole,
        text_outputs: Vec<String>,
        tool_calls: Vec<ToolCallRecord>,
        usage: ProviderUsage,
    },
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
        let timeout_ms = self.policy.for_request(&request).limits.request_timeout_ms;
        tokio::time::timeout(
            Duration::from_millis(timeout_ms.max(1)),
            self.run_inner(request),
        )
        .await
        .map_err(|_| KernelError::RequestTimeout { timeout_ms })?
    }

    async fn run_inner(&self, request: NormalizedRequest) -> Result<KernelOutput, KernelError> {
        let request_policy = self.policy.for_request(&request);
        let scope = request.isolation_key();
        let mut store = ArtifactStore::new();
        let mut ledger = ToolLedger::new();
        let mut text_artifacts = Vec::new();
        let mut encrypted_subagent_state = Vec::new();
        let mut usage = ProviderUsage::default();
        let mut provider_call_count = 0_u32;
        let mut trace_events = Vec::new();
        let system_instructions = system_instructions_with_policy(&request, &request_policy);
        let planner_provider_options =
            planner_provider_options(&request.provider_options, &request_policy);
        let internal_agent_provider_options = internal_agent_provider_options(
            &request.provider_options,
            u64::from(request_policy.limits.max_tokens_per_agent),
        );
        let auxiliary_agent_provider_options = provider_options_with_min_output_tokens(
            &request.provider_options,
            u64::from(request_policy.limits.max_tokens_per_agent),
        );

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
            limits: AgentLimits {
                max_tokens: request_policy.limits.max_tokens_per_agent,
                max_tool_calls: request_policy.limits.max_tool_calls_per_agent,
                timeout_ms: request_policy.limits.agent_timeout_ms,
            },
        };

        let mut graph = TaskGraph::new(root_task_id.clone());
        graph.insert_task(root_task.clone());

        let mut planning_issue = None;

        if request.reasoning_effort.is_direct() {
            trace_events.push(trace_agent_input(&root_task, flatten_text(&request)));
            let root_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: root_task.clone(),
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
                    provider_options: request.provider_options.clone(),
                })
                .await?;
            accumulate_usage(&mut usage, &mut provider_call_count, &root_response);
            trace_events.extend(trace_events_from_response(&root_task, &root_response));

            Self::apply_provider_response(
                &request_policy,
                &scope,
                &mut store,
                &mut ledger,
                &mut graph,
                &mut text_artifacts,
                root_response,
            )?;
        } else {
            let mut root_preface_artifacts = Vec::new();

            if should_run_root_tool_gate(&request)
                && !should_use_model_orchestration(&request, &request_policy)
            {
                trace_events.push(trace_agent_input(&root_task, flatten_text(&request)));
                let root_response = self
                    .invoke_provider(ProviderRequest {
                        scope: scope.clone(),
                        task: root_task.clone(),
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
                        provider_options: request.provider_options.clone(),
                    })
                    .await?;
                accumulate_usage(&mut usage, &mut provider_call_count, &root_response);
                trace_events.extend(trace_events_from_response(&root_task, &root_response));

                Self::apply_provider_response(
                    &request_policy,
                    &scope,
                    &mut store,
                    &mut ledger,
                    &mut graph,
                    &mut root_preface_artifacts,
                    root_response,
                )?;
            }

            let root_has_unresolved_tools = !ledger
                .unresolved_calls_for_task(&scope, &root_task_id)
                .is_empty();

            let graph_has_children = graph
                .tasks
                .values()
                .any(|task| task.parent_task_id.is_some());

            if !root_has_unresolved_tools
                && !graph_has_children
                && should_use_model_orchestration(&request, &request_policy)
            {
                let planner_task = orchestration_planner_task(&root_task_id, &request_policy);
                let planner_input = orchestration_planner_input(&request, &request_policy);
                trace_events.push(trace_agent_input(&planner_task, planner_input.clone()));
                let mut planner_response = self
                    .invoke_provider(ProviderRequest {
                        scope: scope.clone(),
                        task: planner_task.clone(),
                        model: request.model.clone(),
                        system_instructions: system_instructions.clone(),
                        thinking_enabled: request.thinking_enabled,
                        thinking_format: request.thinking_format.clone(),
                        input_text: planner_input,
                        messages: Vec::new(),
                        media_artifacts: request.media_artifacts.clone(),
                        artifacts: Vec::new(),
                        tools: Vec::new(),
                        tool_choice: ToolChoice::None,
                        parallel_tool_calls: None,
                        tool_results: Vec::new(),
                        provider_options: planner_provider_options.clone(),
                    })
                    .await?;
                accumulate_usage(&mut usage, &mut provider_call_count, &planner_response);
                trace_events.extend(trace_events_from_response(&planner_task, &planner_response));
                let mut plan_coverage = planner_coverage(&planner_response);
                let mut repair_attempts = 0_usize;
                while should_repair_orchestration_plan(&plan_coverage, &request_policy)
                    && repair_attempts < MAX_ORCHESTRATION_REPAIR_ATTEMPTS
                {
                    let repair_task =
                        orchestration_planner_repair_task(&root_task_id, &request_policy);
                    let repair_input = orchestration_planner_repair_input(
                        &request,
                        &request_policy,
                        &planner_response,
                        repair_attempts + 1,
                    );
                    trace_events.push(trace_agent_input(&repair_task, repair_input.clone()));
                    let repair_response = self
                        .invoke_provider(ProviderRequest {
                            scope: scope.clone(),
                            task: repair_task.clone(),
                            model: request.model.clone(),
                            system_instructions: system_instructions.clone(),
                            thinking_enabled: request.thinking_enabled,
                            thinking_format: request.thinking_format.clone(),
                            input_text: repair_input,
                            messages: Vec::new(),
                            media_artifacts: request.media_artifacts.clone(),
                            artifacts: Vec::new(),
                            tools: Vec::new(),
                            tool_choice: ToolChoice::None,
                            parallel_tool_calls: None,
                            tool_results: Vec::new(),
                            provider_options: planner_provider_options.clone(),
                        })
                        .await?;
                    accumulate_usage(&mut usage, &mut provider_call_count, &repair_response);
                    trace_events.extend(trace_events_from_response(&repair_task, &repair_response));
                    let repair_coverage = planner_coverage(&repair_response);
                    planner_response = repair_response;
                    plan_coverage = repair_coverage;
                    repair_attempts += 1;
                }
                if should_repair_orchestration_plan(&plan_coverage, &request_policy) {
                    return Err(KernelError::ProviderRejected(format!(
                        "model orchestration returned {} child agent(s); reasoning effort {} requires exactly {} child agent(s); refusing root-only fallback",
                        plan_coverage.child_count,
                        reasoning_effort_name(&request.reasoning_effort),
                        request_policy.limits.max_agents_per_request
                    )));
                }
                let plan_seen = Self::apply_spawn_plan_response(
                    &request_policy,
                    &scope,
                    &mut store,
                    &mut graph,
                    planner_response,
                )?;
                if !plan_seen {
                    planning_issue = Some(VerificationIssue {
                        code: "missing_model_spawn_plan".to_string(),
                        message: "model orchestration did not return a valid structured spawn_plan"
                            .to_string(),
                    });
                }
            }

            let graph_has_children = graph
                .tasks
                .values()
                .any(|task| task.parent_task_id.is_some());
            if !root_has_unresolved_tools && !graph_has_children {
                if root_preface_artifacts.is_empty() {
                    trace_events.push(trace_agent_input(&root_task, flatten_text(&request)));
                    let root_response = self
                        .invoke_provider(ProviderRequest {
                            scope: scope.clone(),
                            task: root_task.clone(),
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
                            provider_options: request.provider_options.clone(),
                        })
                        .await?;
                    accumulate_usage(&mut usage, &mut provider_call_count, &root_response);
                    trace_events.extend(trace_events_from_response(&root_task, &root_response));
                    Self::apply_provider_response(
                        &request_policy,
                        &scope,
                        &mut store,
                        &mut ledger,
                        &mut graph,
                        &mut text_artifacts,
                        root_response,
                    )?;
                } else {
                    text_artifacts.extend(root_preface_artifacts);
                }
            }
        }

        let tasks: Vec<SubtaskSpec> = graph
            .tasks
            .values()
            .filter(|task| task.parent_task_id.is_some())
            .cloned()
            .collect();
        let has_child_tasks = !tasks.is_empty();

        let child_requests = tasks
            .into_iter()
            .enumerate()
            .map(|(index, mut task)| {
                task.limits.timeout_ms = request_policy.limits.agent_timeout_ms;
                let artifacts = task
                    .input_artifact_refs
                    .iter()
                    .filter_map(|artifact_ref| {
                        store
                            .get(&artifact_ref.scope, &artifact_ref.artifact_id)
                            .cloned()
                    })
                    .collect();
                let input_text = child_agent_input(&request, &task);
                let request = ProviderRequest {
                    scope: scope.clone(),
                    task: task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.clone(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: input_text.clone(),
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts,
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                    provider_options: internal_agent_provider_options.clone(),
                };
                (index, task, input_text, request)
            })
            .collect::<Vec<_>>();

        let parallel_limit = usize::from(request_policy.limits.max_parallel_agents.max(1));
        let child_results = stream::iter(child_requests.into_iter().map(
            |(index, task, input_text, request)| async move {
                let response = self.invoke_provider(request).await?;
                Ok::<_, KernelError>((index, task, input_text, response))
            },
        ))
        .buffer_unordered(parallel_limit)
        .collect::<Vec<_>>()
        .await;
        let mut child_results = child_results
            .into_iter()
            .collect::<Result<Vec<_>, KernelError>>()?;
        child_results.sort_by_key(|(index, _, _, _)| *index);
        for child_result in child_results {
            let (_, task, input_text, child_response) = child_result;
            accumulate_usage(&mut usage, &mut provider_call_count, &child_response);
            trace_events.push(trace_agent_input(&task, input_text));
            trace_events.extend(trace_events_from_response(&task, &child_response));
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

        let mut root_unresolved: Vec<ToolCallId> = ledger
            .unresolved_calls_for_task(&scope, &root_task_id)
            .into_iter()
            .map(|call| call.tool_call_id.clone())
            .collect();

        let mut public_tool_calls: Vec<ToolCallRecord> = ledger
            .unresolved_calls_for_task(&scope, &root_task_id)
            .into_iter()
            .cloned()
            .collect();

        let mut root_continuation_artifacts = Vec::new();
        if root_unresolved.is_empty() && has_child_tasks && root_tools_available(&request) {
            let root_continuation_task = SubtaskSpec {
                objective: "continue root-visible tool execution from worker findings, or produce a final answer only when complete".to_string(),
                ..root_task.clone()
            };
            let continuation_input = root_tool_continuation_input(&request, &text_artifacts);
            trace_events.push(trace_agent_input(
                &root_continuation_task,
                continuation_input.clone(),
            ));
            let continuation_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: root_continuation_task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.clone(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: continuation_input,
                    messages: request.messages.clone(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: text_artifacts.clone(),
                    tools: request.tools.clone(),
                    tool_choice: request.tool_choice.clone(),
                    parallel_tool_calls: request.parallel_tool_calls,
                    tool_results: request.tool_results.clone(),
                    provider_options: request.provider_options.clone(),
                })
                .await?;
            accumulate_usage(&mut usage, &mut provider_call_count, &continuation_response);
            trace_events.extend(trace_events_from_response(
                &root_continuation_task,
                &continuation_response,
            ));
            Self::apply_provider_response(
                &request_policy,
                &scope,
                &mut store,
                &mut ledger,
                &mut graph,
                &mut root_continuation_artifacts,
                continuation_response,
            )?;

            root_unresolved = ledger
                .unresolved_calls_for_task(&scope, &root_task_id)
                .into_iter()
                .map(|call| call.tool_call_id.clone())
                .collect();
            public_tool_calls = ledger
                .unresolved_calls_for_task(&scope, &root_task_id)
                .into_iter()
                .cloned()
                .collect();
        }

        if request.public_reasoning_enabled && root_unresolved.is_empty() && has_child_tasks {
            let summary_task = SubtaskSpec {
                task_id: TaskId::from("reasoning-summary"),
                parent_task_id: Some(root_task_id.clone()),
                spawn_depth: 0,
                role: AgentRole::ReasoningSummarizer,
                objective: "summarize bounded worker-agent outputs for public reasoning"
                    .to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits {
                    max_tokens: request_policy.limits.max_tokens_per_agent,
                    max_tool_calls: 0,
                    timeout_ms: request_policy.limits.agent_timeout_ms,
                },
            };
            graph.insert_task(summary_task.clone());
            let summary_input = reasoning_summary_input(&request, &trace_events);
            trace_events.push(trace_agent_input(&summary_task, summary_input.clone()));
            let summary_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: summary_task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.clone(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: summary_input,
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: text_artifacts.clone(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                    provider_options: auxiliary_agent_provider_options.clone(),
                })
                .await?;
            accumulate_usage(&mut usage, &mut provider_call_count, &summary_response);
            trace_events.extend(trace_events_from_response(&summary_task, &summary_response));
        }

        let final_text = if root_unresolved.is_empty() && !root_continuation_artifacts.is_empty() {
            synthesize_text(&root_continuation_artifacts)
        } else if root_unresolved.is_empty() && has_child_tasks {
            let synth_task = SubtaskSpec {
                task_id: TaskId::from("synthesizer"),
                parent_task_id: Some(root_task_id.clone()),
                spawn_depth: 0,
                role: AgentRole::Synthesizer,
                objective: "synthesize a natural final answer for the user".to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits {
                    max_tokens: request_policy.limits.max_tokens_per_agent,
                    max_tool_calls: 0,
                    timeout_ms: request_policy.limits.agent_timeout_ms,
                },
            };
            graph.insert_task(synth_task.clone());
            let synth_input = synthesis_input(&request, &text_artifacts);
            trace_events.push(trace_agent_input(&synth_task, synth_input.clone()));
            let synth_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: synth_task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.clone(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: synth_input,
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: Vec::new(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                    provider_options: request.provider_options.clone(),
                })
                .await?;
            accumulate_usage(&mut usage, &mut provider_call_count, &synth_response);
            trace_events.extend(trace_events_from_response(&synth_task, &synth_response));
            synthesize_text(&synth_response.artifacts)
        } else if root_unresolved.is_empty() {
            synthesize_text(&text_artifacts)
        } else {
            String::new()
        };
        let mut final_text = if root_unresolved.is_empty() && !final_text.trim().is_empty() {
            normalize_structured_text_layout(&final_text)
        } else {
            final_text
        };

        let semantic_verification = if request_policy.semantic_verification.enabled
            && !request.reasoning_effort.is_direct()
            && root_unresolved.is_empty()
            && !final_text.trim().is_empty()
        {
            self.run_semantic_verification(
                &request,
                &request_policy,
                &scope,
                &root_task_id,
                &system_instructions,
                &text_artifacts,
                &mut graph,
                &mut usage,
                &mut provider_call_count,
                &mut trace_events,
                final_text,
            )
            .await
        } else {
            SemanticVerificationResult {
                final_text,
                passed: true,
                issues: Vec::new(),
                covered_artifact_ids: text_artifacts
                    .iter()
                    .map(|artifact| artifact.id().as_ref().to_string())
                    .collect(),
            }
        };
        final_text = semantic_verification.final_text;

        let token_used = usage.input_tokens.saturating_add(usage.output_tokens);
        let mut issues = semantic_verification.issues;
        if !root_unresolved.is_empty() {
            issues.push(VerificationIssue {
                code: "unresolved_tool_calls".to_string(),
                message: "tool calls must be resolved before final synthesis".to_string(),
            });
        }
        if let Some(issue) = planning_issue {
            issues.push(issue);
        }
        let passed = semantic_verification.passed
            && !issues
                .iter()
                .any(|issue| issue.code != "missing_model_spawn_plan");
        let verification = VerificationReport {
            request_id: request.request_id.clone(),
            passed,
            issues,
            artifact_coverage: text_artifacts
                .iter()
                .map(|artifact| ArtifactCoverage {
                    artifact_id: artifact.id(),
                    kind: ArtifactKind::Text,
                    covered: semantic_verification
                        .covered_artifact_ids
                        .contains(artifact.id().as_ref()),
                })
                .collect(),
            unresolved_tool_calls: root_unresolved,
            budget_summary: BudgetSummary {
                token_budget: request_policy.limits.max_total_tokens,
                token_used,
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
            usage,
            provider_call_count,
            trace_events,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_semantic_verification(
        &self,
        request: &NormalizedRequest,
        policy: &KernelPolicy,
        scope: &IsolationKey,
        root_task_id: &TaskId,
        system_instructions: &[String],
        evidence: &[AgentArtifact],
        graph: &mut TaskGraph,
        usage: &mut ProviderUsage,
        provider_call_count: &mut u32,
        trace_events: &mut Vec<KernelTraceEvent>,
        mut candidate: String,
    ) -> SemanticVerificationResult {
        let mut repair_attempts = 0_u8;
        loop {
            let verifier_task = SubtaskSpec {
                task_id: TaskId::from(format!("semantic-verifier-{repair_attempts}")),
                parent_task_id: Some(root_task_id.clone()),
                spawn_depth: 0,
                role: AgentRole::Verifier,
                objective: "independently verify the candidate answer against the user request and worker evidence".to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Verification],
                allowed_capabilities: CapabilitySet::from([Capability::Text, Capability::Verify]),
                limits: AgentLimits {
                    max_tokens: policy.limits.max_tokens_per_agent,
                    max_tool_calls: 0,
                    timeout_ms: policy.limits.agent_timeout_ms,
                },
            };
            graph.insert_task(verifier_task.clone());
            let verifier_input = semantic_verifier_input(request, evidence, &candidate);
            trace_events.push(trace_agent_input(&verifier_task, verifier_input.clone()));
            let verifier_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: verifier_task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.to_vec(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: verifier_input,
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: evidence.to_vec(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                    provider_options: provider_options_with_min_output_tokens(
                        &request.provider_options,
                        u64::from(policy.limits.max_tokens_per_agent),
                    ),
                })
                .await;
            let verifier_response = match verifier_response {
                Ok(response) => response,
                Err(error) => {
                    return SemanticVerificationResult {
                        final_text: candidate,
                        passed: false,
                        issues: vec![VerificationIssue {
                            code: "semantic_verifier_error".to_string(),
                            message: error.to_string(),
                        }],
                        covered_artifact_ids: BTreeSet::new(),
                    };
                }
            };
            accumulate_usage(usage, provider_call_count, &verifier_response);
            trace_events.extend(trace_events_from_response(
                &verifier_task,
                &verifier_response,
            ));
            let verdict = match semantic_verdict_from_response(&verifier_response) {
                Some(verdict) => verdict,
                None => {
                    return SemanticVerificationResult {
                        final_text: candidate,
                        passed: false,
                        issues: vec![VerificationIssue {
                            code: "invalid_semantic_verdict".to_string(),
                            message: "semantic verifier did not return the required JSON verdict"
                                .to_string(),
                        }],
                        covered_artifact_ids: BTreeSet::new(),
                    };
                }
            };
            let covered_artifact_ids = verdict
                .covered_artifact_ids
                .into_iter()
                .collect::<BTreeSet<_>>();
            if verdict.passed {
                return SemanticVerificationResult {
                    final_text: candidate,
                    passed: true,
                    issues: Vec::new(),
                    covered_artifact_ids,
                };
            }
            let issues = if verdict.issues.is_empty() {
                vec![VerificationIssue {
                    code: "semantic_verification_failed".to_string(),
                    message: "semantic verifier rejected the candidate without issue details"
                        .to_string(),
                }]
            } else {
                verdict.issues
            };
            if repair_attempts >= policy.semantic_verification.max_repair_attempts {
                return SemanticVerificationResult {
                    final_text: candidate,
                    passed: false,
                    issues,
                    covered_artifact_ids,
                };
            }

            repair_attempts = repair_attempts.saturating_add(1);
            let repair_task = SubtaskSpec {
                task_id: TaskId::from(format!("semantic-repair-{repair_attempts}")),
                parent_task_id: Some(root_task_id.clone()),
                spawn_depth: 0,
                role: AgentRole::Synthesizer,
                objective: "repair the candidate answer using bounded verifier feedback and existing evidence".to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits {
                    max_tokens: policy.limits.max_tokens_per_agent,
                    max_tool_calls: 0,
                    timeout_ms: policy.limits.agent_timeout_ms,
                },
            };
            graph.insert_task(repair_task.clone());
            let repair_input = semantic_repair_input(request, evidence, &candidate, &issues);
            trace_events.push(trace_agent_input(&repair_task, repair_input.clone()));
            let repair_response = self
                .invoke_provider(ProviderRequest {
                    scope: scope.clone(),
                    task: repair_task.clone(),
                    model: request.model.clone(),
                    system_instructions: system_instructions.to_vec(),
                    thinking_enabled: request.thinking_enabled,
                    thinking_format: request.thinking_format.clone(),
                    input_text: repair_input,
                    messages: Vec::new(),
                    media_artifacts: request.media_artifacts.clone(),
                    artifacts: evidence.to_vec(),
                    tools: Vec::new(),
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    tool_results: Vec::new(),
                    provider_options: request.provider_options.clone(),
                })
                .await;
            let repair_response = match repair_response {
                Ok(response) => response,
                Err(error) => {
                    let mut issues = issues;
                    issues.push(VerificationIssue {
                        code: "semantic_repair_error".to_string(),
                        message: error.to_string(),
                    });
                    return SemanticVerificationResult {
                        final_text: candidate,
                        passed: false,
                        issues,
                        covered_artifact_ids,
                    };
                }
            };
            accumulate_usage(usage, provider_call_count, &repair_response);
            trace_events.extend(trace_events_from_response(&repair_task, &repair_response));
            let repaired = synthesize_text(&repair_response.artifacts);
            if repaired.trim().is_empty() {
                let mut issues = issues;
                issues.push(VerificationIssue {
                    code: "empty_semantic_repair".to_string(),
                    message: "semantic repair returned no candidate answer".to_string(),
                });
                return SemanticVerificationResult {
                    final_text: candidate,
                    passed: false,
                    issues,
                    covered_artifact_ids,
                };
            }
            candidate = normalize_structured_text_layout(&repaired);
        }
    }

    async fn invoke_provider(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, KernelError> {
        let task_id = request.task.task_id.clone();
        let timeout_ms = if request.task.limits.timeout_ms == 0 {
            AgentLimits::default().timeout_ms
        } else {
            request.task.limits.timeout_ms
        };
        tokio::time::timeout(
            Duration::from_millis(timeout_ms.max(1)),
            self.provider.invoke(request),
        )
        .await
        .map_err(|_| KernelError::AgentTimeout {
            task_id,
            timeout_ms,
        })?
        .map_err(KernelError::from)
    }

    pub async fn stream_root(
        &self,
        request: NormalizedRequest,
    ) -> Result<ProviderStream, KernelError> {
        let request_policy = self.policy.for_request(&request);
        let scope = request.isolation_key();
        let mut root_task = root_task(&scope, &request);
        root_task.limits.timeout_ms = request_policy.limits.agent_timeout_ms;
        let task_id = root_task.task_id.clone();
        let timeout_ms = root_task.limits.timeout_ms;
        tokio::time::timeout(
            Duration::from_millis(timeout_ms.max(1)),
            self.provider.stream(ProviderRequest {
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
                provider_options: request.provider_options.clone(),
            }),
        )
        .await
        .map_err(|_| KernelError::AgentTimeout {
            task_id,
            timeout_ms,
        })?
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

    fn apply_spawn_plan_response(
        policy: &KernelPolicy,
        scope: &IsolationKey,
        store: &mut ArtifactStore,
        graph: &mut TaskGraph,
        response: ProviderResponse,
    ) -> Result<bool, KernelError> {
        let mut saw_plan = false;
        let mut plans = Vec::new();
        for artifact in response.artifacts {
            match artifact {
                AgentArtifact::SpawnPlan { plan, .. } => {
                    saw_plan = true;
                    plans.push(plan);
                }
                other => {
                    store.insert(other);
                }
            }
        }
        for plan in plans {
            SpawnValidator::new(policy.clone()).validate_and_apply(scope, store, graph, &plan)?;
        }
        Ok(saw_plan)
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

        if request.task.role == AgentRole::Leader
            && request.task.objective.contains("orchestration plan")
        {
            return mock_spawn_plan_response(request);
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
            return mock_spawn_plan_response(request);
        }

        let text = match request.task.role {
            AgentRole::Worker
                if request
                    .input_text
                    .contains("Root-visible tool context observed so far:") =>
            {
                format!(
                    "child completed: {}\n{}",
                    request.task.objective, request.input_text
                )
            }
            AgentRole::Worker => format!("child completed: {}", request.task.objective),
            AgentRole::ReasoningSummarizer => mock_reasoning_summary(&request.input_text),
            AgentRole::Verifier => serde_json::json!({
                "passed": true,
                "issues": [],
                "covered_artifact_ids": request
                    .artifacts
                    .iter()
                    .map(|artifact| artifact.id().as_ref().to_string())
                    .collect::<Vec<_>>()
            })
            .to_string(),
            AgentRole::Synthesizer => mock_synthesis_text(&request.input_text),
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

fn mock_spawn_plan_response(request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
    let target = target_child_agents(&request);
    let image_ref: Vec<ArtifactRef> = request
        .media_artifacts
        .first()
        .map(|media| ArtifactRef {
            scope: request.scope.clone(),
            artifact_id: media.id.clone(),
        })
        .into_iter()
        .collect();
    let children = (0..target)
        .map(|index| SubtaskSpec {
            task_id: TaskId::from(format!("child-{index:02}")),
            parent_task_id: Some(TaskId::from("root")),
            spawn_depth: 1,
            role: AgentRole::Worker,
            objective: format!("child visual inspection slice {index}"),
            input_artifact_refs: image_ref.clone(),
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([Capability::Text, Capability::Image]),
            limits: AgentLimits::default(),
        })
        .collect();
    Ok(ProviderResponse {
        artifacts: vec![AgentArtifact::SpawnPlan {
            id: ArtifactId::from("spawn-plan-1"),
            scope: request.scope.clone(),
            plan: SpawnPlan {
                parent_task_id: TaskId::from("root"),
                reason: "Need bounded child inspection".to_string(),
                children,
                expected_artifacts: vec![ArtifactKind::Text],
                budget_request: BudgetRequest {
                    max_tokens: 256,
                    max_tool_calls: 0,
                },
            },
        }],
        tool_calls: Vec::new(),
        usage: Default::default(),
    })
}

fn mock_synthesis_text(input_text: &str) -> String {
    if input_text.contains("Root-visible tool context observed so far:") {
        return format!(
            "Here is a clear, usable answer based on the verified agent results.\n{}",
            input_text
        );
    }
    input_text
        .lines()
        .find(|line| line.contains("Here is a clear, usable answer from tool results:"))
        .map(str::to_string)
        .unwrap_or_else(|| {
            "Here is a clear, usable answer based on the verified agent results.".to_string()
        })
}

fn mock_reasoning_summary(input_text: &str) -> String {
    let mut lines = Vec::new();
    for line in input_text.lines() {
        let trimmed = line.trim();
        if let Some((task_id, _)) = trimmed.split_once(" (Worker agent):") {
            lines.push(format!(
                "{task_id} (Worker agent): summarized worker finding"
            ));
        }
    }

    if lines.is_empty() {
        "No worker summaries were available.".to_string()
    } else {
        lines.join("\n")
    }
}

fn flatten_text(request: &NormalizedRequest) -> String {
    request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(text.clone()),
            NormalizedContentPart::ProviderContent { value, .. } => {
                Some(provider_content_prompt_text(value))
            }
            NormalizedContentPart::ToolCall { .. } => None,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn provider_content_prompt_text(value: &serde_json::Value) -> String {
    if let Some(text) = value
        .get("text")
        .or_else(|| value.get("thinking"))
        .and_then(serde_json::Value::as_str)
    {
        return text.to_string();
    }
    let kind = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("provider_content");
    format!("[{kind} content attached]")
}

fn system_instructions(request: &NormalizedRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|message| message.role == MessageRole::System)
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(text.clone()),
            NormalizedContentPart::ProviderContent { value, .. } => {
                Some(provider_content_prompt_text(value))
            }
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
    if !request.tools.is_empty() {
        instructions.push(
            "Tool execution policy: child agents cannot call client tools. When the task needs \
             commands, file edits, retrieval, or more observation, the root leader must emit the \
             next root-visible tool call instead of finalizing with a plan. \
             Continue the tool loop until the requested work is actually complete. Once accepted \
             tool results show the requested artifacts were created and at least one relevant \
             validation succeeded, stop calling tools and return the final answer. Do not repeat \
             successful commands or equivalent validation just to gather more confidence. For \
             environment-building tasks, planning/status-only tools are not a substitute for \
             concrete execution or observation tools. Prefer a cohesive command/script that safely \
             performs related setup and verification steps together over many one-step tool calls."
                .to_string(),
        );
        if let Some(tool_policy) = stateful_tool_policy_instruction(request) {
            instructions.push(tool_policy);
        }
    }
    instructions
}

fn stateful_tool_policy_instruction(request: &NormalizedRequest) -> Option<String> {
    let stateful_tools = request
        .tools
        .iter()
        .filter_map(stateful_tool_contract)
        .collect::<Vec<_>>();
    if stateful_tools.is_empty() {
        return None;
    }

    let handles = observed_stateful_handles(request);
    let handle_text = if handles.is_empty() {
        "No state handle IDs have been observed in accepted tool results for this conversation."
            .to_string()
    } else {
        format!(
            "Observed state handle IDs from accepted tool results: {}.",
            handles.into_iter().collect::<Vec<_>>().join(", ")
        )
    };

    Some(format!(
        "Stateful tool policy: tools with session/process handle parameters may only be called \
         with handle IDs explicitly returned by accepted tool results in this same conversation. \
         Do not invent numeric IDs, reuse IDs from unrelated turns, or call a state-continuation \
         tool when the referenced operation already completed without returning a live handle. \
         If no active handle is visible, use an appropriate stateless/starter tool instead.\n\
         {handle_text}\n{}",
        stateful_tools.join("\n")
    ))
}

fn stateful_tool_contract(tool: &ToolDefinition) -> Option<String> {
    let properties = tool
        .input_schema
        .get("properties")
        .and_then(|value| value.as_object())?;
    let state_fields = properties
        .keys()
        .filter(|name| is_state_handle_field(name))
        .cloned()
        .collect::<Vec<_>>();
    if state_fields.is_empty() {
        return None;
    }

    let required = tool
        .input_schema
        .get("required")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let required_state_fields = state_fields
        .iter()
        .filter(|name| required.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let required_text = if required_state_fields.is_empty() {
        String::new()
    } else {
        format!(
            " Required handle field(s): {}.",
            required_state_fields.join(", ")
        )
    };

    Some(format!(
        "- Tool `{}` has state handle field(s): {}.{}",
        tool.name,
        state_fields.join(", "),
        required_text
    ))
}

fn observed_stateful_handles(request: &NormalizedRequest) -> BTreeSet<String> {
    let mut handles = BTreeSet::new();
    for result in request
        .tool_results
        .iter()
        .filter(|result| result.status == ToolResultStatus::Accepted)
    {
        collect_stateful_handles(&result.result_json, &mut handles);
    }
    handles
}

fn collect_stateful_handles(value: &serde_json::Value, handles: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                if is_state_handle_field(key)
                    && let Some(handle) = state_handle_value(value)
                {
                    handles.insert(format!("{key}={handle}"));
                }
                collect_stateful_handles(value, handles);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_stateful_handles(value, handles);
            }
        }
        _ => {}
    }
}

fn state_handle_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.trim().is_empty() => {
            Some(value.trim().to_string())
        }
        serde_json::Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn is_state_handle_field(name: &str) -> bool {
    let normalized = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "sessionid" | "processid" | "procid" | "pid" | "targetsessionid" | "targetprocessid"
    )
}

fn orchestration_policy_instruction(effort: &ReasoningEffort, policy: &KernelPolicy) -> String {
    let max_agents = policy.limits.max_agents_per_request;
    format!(
        "Orchestration policy: reasoning_effort={}; max_agents_per_request={max_agents}; max_parallel_agents={}; target_parallel_agents={max_agents}; max_spawn_depth={}; max_total_tool_calls={}; token_accounting_reference={}. Multi-agent decomposition must be model-selected from the actual task context, not a generic template. For non-direct reasoning, target_parallel_agents is the required child-agent count; empty or under-target spawn plans are rejected instead of falling back to root-only execution. max_parallel_agents is only the concurrent backend request limit, not the child-agent count. Token accounting is telemetry only and must not suppress child-agent execution or reduce final answer quality. Low should stay compact; medium/high/xhigh should broaden coverage and verification. Only root-visible tool calls and the final synthesis are public.",
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

fn should_run_root_tool_gate(request: &NormalizedRequest) -> bool {
    root_tools_available(request) && request.tool_results.is_empty()
}

fn root_tools_available(request: &NormalizedRequest) -> bool {
    !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None)
}

fn should_use_model_orchestration(request: &NormalizedRequest, policy: &KernelPolicy) -> bool {
    !request.reasoning_effort.is_direct() && policy.limits.max_agents_per_request > 1
}

fn target_child_agents(request: &ProviderRequest) -> usize {
    request
        .task
        .objective
        .split("exactly ")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            request
                .system_instructions
                .iter()
                .flat_map(|instruction| instruction.split([';', '\n']))
                .filter_map(|part| part.trim().strip_prefix("target_parallel_agents="))
                .filter_map(|value| value.parse::<usize>().ok())
                .next()
        })
        .unwrap_or(1)
        .max(1)
}

fn provider_options_with_min_output_tokens(
    options: &serde_json::Value,
    min_tokens: u64,
) -> serde_json::Value {
    let mut options = options.clone();
    let Some(object) = options.as_object_mut() else {
        return options;
    };

    let mut saw_limit = false;
    for key in OUTPUT_LIMIT_KEYS {
        if object.contains_key(key) {
            saw_limit = true;
            raise_output_token_limit(object, key, min_tokens);
        }
    }
    if !saw_limit {
        object.insert("max_tokens".to_string(), json!(min_tokens));
    }

    options
}

const OUTPUT_LIMIT_KEYS: [&str; 5] = [
    "max_tokens",
    "max_completion_tokens",
    "max_new_tokens",
    "n_predict",
    "num_predict",
];

fn internal_agent_provider_options(
    options: &serde_json::Value,
    min_tokens: u64,
) -> serde_json::Value {
    let mut options = strip_internal_stop_conditions(options);
    options = provider_options_with_min_output_tokens(&options, min_tokens);
    options
}

fn strip_internal_stop_conditions(options: &serde_json::Value) -> serde_json::Value {
    let mut options = options.clone();
    if let Some(object) = options.as_object_mut() {
        object.remove("stop");
        object.remove("stopping_strings");
    }
    options
}

fn planner_provider_options(
    options: &serde_json::Value,
    policy: &KernelPolicy,
) -> serde_json::Value {
    let mut options = internal_agent_provider_options(options, planner_output_token_floor(policy));
    if let Some(object) = options.as_object_mut() {
        object.insert("temperature".to_string(), json!(0));
        object.insert(
            "response_format".to_string(),
            json!({
                "type": "json_object"
            }),
        );
    }
    options
}

fn raise_output_token_limit(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    min_tokens: u64,
) {
    let should_raise = match object.get(key) {
        Some(serde_json::Value::Number(number)) => {
            number.as_u64().is_some_and(|value| value < min_tokens)
        }
        Some(serde_json::Value::String(text)) => text
            .trim()
            .parse::<u64>()
            .is_ok_and(|value| value < min_tokens),
        Some(serde_json::Value::Null) | None => true,
        _ => false,
    };
    if should_raise {
        object.insert(key.to_string(), json!(min_tokens));
    }
}

fn planner_output_token_floor(policy: &KernelPolicy) -> u64 {
    u64::from(policy.limits.max_tokens_per_agent)
        .max(u64::from(policy.limits.max_agents_per_request).saturating_mul(128))
}

#[derive(Debug, Clone, Copy)]
struct PlannerCoverage {
    saw_plan: bool,
    child_count: usize,
}

fn planner_coverage(response: &ProviderResponse) -> PlannerCoverage {
    let mut saw_plan = false;
    let mut child_count = 0_usize;
    for artifact in &response.artifacts {
        if let AgentArtifact::SpawnPlan { plan, .. } = artifact {
            saw_plan = true;
            child_count = child_count.saturating_add(plan.children.len());
        }
    }
    PlannerCoverage {
        saw_plan,
        child_count,
    }
}

fn should_repair_orchestration_plan(coverage: &PlannerCoverage, policy: &KernelPolicy) -> bool {
    let target = usize::from(policy.limits.max_agents_per_request);
    target > 1 && (!coverage.saw_plan || coverage.child_count < target)
}

fn orchestration_planner_task(root_task_id: &TaskId, policy: &KernelPolicy) -> SubtaskSpec {
    SubtaskSpec {
        task_id: root_task_id.clone(),
        parent_task_id: None,
        spawn_depth: 0,
        role: AgentRole::Leader,
        objective: format!(
            "produce a bounded model-selected orchestration plan with exactly {} child agents",
            policy.limits.max_agents_per_request
        ),
        input_artifact_refs: Vec::new(),
        expected_outputs: vec![ArtifactKind::SpawnPlan],
        allowed_capabilities: CapabilitySet::from([
            Capability::Text,
            Capability::Image,
            Capability::Spawn,
        ]),
        limits: AgentLimits {
            max_tokens: policy.limits.max_tokens_per_agent,
            max_tool_calls: 0,
            timeout_ms: policy.limits.agent_timeout_ms,
        },
    }
}

fn orchestration_planner_repair_task(root_task_id: &TaskId, policy: &KernelPolicy) -> SubtaskSpec {
    SubtaskSpec {
        objective: format!(
            "repair a bounded model-selected orchestration plan with exactly {} child agents",
            policy.limits.max_agents_per_request
        ),
        ..orchestration_planner_task(root_task_id, policy)
    }
}

fn orchestration_planner_input(request: &NormalizedRequest, policy: &KernelPolicy) -> String {
    let context = agent_visible_context(request);
    let media_note = if request.media_artifacts.is_empty() {
        "No image artifacts are attached.".to_string()
    } else {
        format!(
            "{} image artifact(s) are attached; assign visual inspection only when needed.",
            request.media_artifacts.len()
        )
    };
    format!(
        "You are the commercial orchestration planner for this single API request.\n\
         Decide the child-agent division from the actual task, tool results, media, and user constraints.\n\
         Return ONLY valid JSON. Do not write prose, markdown, questions, or a final answer.\n\
         This request is not direct mode: children length MUST be exactly {max_agents}. Empty or fewer children is an invalid API contract.\n\
         The model decides each worker's task-specific objective, but the kernel enforces the configured child-agent count.\n\
         Child objectives must be task-specific, non-overlapping, outcome-oriented, and directly useful for final synthesis.\n\
         Keep each child objective short, ideally 12 words or fewer, so the spawn_plan remains compact.\n\
         Avoid generic objectives such as \"analyze the request\" unless that is the actual work product.\n\
         If client tools are needed, assign workers to analyze current evidence and propose next root-visible tool actions; workers cannot call client tools.\n\
         Do not ask the user whether to proceed; assign workers to make progress and verify.\n\n\
         Hard limits enforced by the kernel:\n\
         - parent_task_id must be \"root\"\n\
         - children length must be exactly {max_agents}\n\
         - max_parallel_agents={max_parallel}; this is concurrency only, not the number of children\n\
         - child agents cannot call tools; the kernel fills worker/text-only defaults when optional child fields are omitted\n\n\
         Return this compact JSON shape. The children array shown must contain exactly {max_agents} objects:\n\
         {{\n\
           \"type\": \"spawn_plan\",\n\
           \"plan\": {{\n\
             \"parent_task_id\": \"root\",\n\
             \"reason\": \"why these child agents improve the answer\",\n\
             \"children\": [\n\
               {{\"task_id\":\"short-stable-kebab-case-id-01\",\"objective\":\"specific work this child must complete\"}}\n\
             ]\n\
           }}\n\
         }}\n\n\
         Media context: {media_note}\n\n\
         Request context:\n{context}",
        max_agents = policy.limits.max_agents_per_request,
        max_parallel = policy.limits.max_parallel_agents,
    )
}

fn orchestration_planner_repair_input(
    request: &NormalizedRequest,
    policy: &KernelPolicy,
    previous_response: &ProviderResponse,
    attempt: usize,
) -> String {
    let previous = planner_attempt_summary(previous_response);
    let context = agent_visible_context(request);
    format!(
        "You are repairing the orchestration plan for this same single API request.\n\
         Repair attempt: {attempt}/{max_attempts}.\n\
         Previous orchestration attempt:\n{previous}\n\n\
         The previous attempt was missing, empty, or below the configured coverage tier.\n\
         Return ONLY valid JSON for one complete replacement spawn_plan. The kernel will apply only this repaired plan.\n\
         Keep the division model-selected and task-specific; do not use generic template workers.\n\
         Keep each child objective short, ideally 12 words or fewer.\n\
         target_child_agents={target}; max_agents_per_request={target}; max_parallel_agents={max_parallel}.\n\
         children length MUST be exactly {target}. Empty or fewer children is invalid and will be rejected; do not answer with root-only work.\n\
         Child agents cannot call client tools; if tools are needed, assign workers to analyze evidence and let the root leader make root-visible tool calls later.\n\n\
         Required compact JSON shape: {{\"type\":\"spawn_plan\",\"plan\":{{\"parent_task_id\":\"root\",\"reason\":\"...\",\"children\":[{{\"task_id\":\"child-01\",\"objective\":\"specific worker objective\"}}]}}}}. The children array must contain exactly {target} objects.\n\n\
         Request context:\n{context}",
        target = policy.limits.max_agents_per_request,
        max_parallel = policy.limits.max_parallel_agents,
        max_attempts = MAX_ORCHESTRATION_REPAIR_ATTEMPTS,
    )
}

fn planner_attempt_summary(response: &ProviderResponse) -> String {
    let mut lines = Vec::new();
    for artifact in &response.artifacts {
        match artifact {
            AgentArtifact::SpawnPlan { plan, .. } => {
                lines.push(format!(
                    "spawn_plan reason: {}; children: {}",
                    plan.reason,
                    plan.children.len()
                ));
                for child in &plan.children {
                    lines.push(format!("- {}: {}", child.task_id.as_ref(), child.objective));
                }
            }
            AgentArtifact::Text { text, .. } => {
                let text = text.trim();
                if !text.is_empty() {
                    lines.push(format!("non-JSON planner text: {text}"));
                }
            }
            _ => {}
        }
    }
    if lines.is_empty() {
        "No spawn_plan artifact was returned.".to_string()
    } else {
        lines.join("\n")
    }
}

fn semantic_verifier_input(
    request: &NormalizedRequest,
    evidence: &[AgentArtifact],
    candidate: &str,
) -> String {
    format!(
        "Act as an independent semantic verifier. Check factual consistency, instruction coverage, contradictions, unsupported claims, and whether the candidate directly answers the user. Use only the root-visible request and evidence below. Do not repair the answer. Return ONLY one JSON object with this exact shape: {json_shape}. Every issue must have a stable short code and actionable message. covered_artifact_ids must list only evidence IDs actually checked.\n\nRoot-visible request:\n{request_text}\n\nEvidence:\n{evidence_text}\n\nCandidate answer:\n{candidate}",
        json_shape = r#"{"passed":true,"issues":[{"code":"...","message":"..."}],"covered_artifact_ids":["..."]}"#,
        request_text = flatten_text(request),
        evidence_text = semantic_evidence_text(evidence),
    )
}

fn semantic_repair_input(
    request: &NormalizedRequest,
    evidence: &[AgentArtifact],
    candidate: &str,
    issues: &[VerificationIssue],
) -> String {
    format!(
        "Repair the candidate answer using only the root-visible request, existing evidence, and verifier issues below. Resolve every issue without inventing evidence. Preserve the caller's output format, stop conditions, and requested level of detail. Return only the complete replacement answer.\n\nRoot-visible request:\n{request_text}\n\nEvidence:\n{evidence_text}\n\nVerifier issues:\n{issues}\n\nCandidate answer:\n{candidate}",
        request_text = flatten_text(request),
        evidence_text = semantic_evidence_text(evidence),
        issues = serde_json::to_string(issues).unwrap_or_else(|_| "[]".to_string()),
    )
}

fn semantic_evidence_text(evidence: &[AgentArtifact]) -> String {
    let rendered = evidence
        .iter()
        .filter_map(|artifact| match artifact {
            AgentArtifact::Text { id, text, .. } => {
                Some(format!("artifact_id={}\n{}", id.as_ref(), text))
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    if rendered.is_empty() {
        "(no worker text artifacts)".to_string()
    } else {
        rendered
    }
}

fn semantic_verdict_from_response(response: &ProviderResponse) -> Option<SemanticVerdict> {
    response.artifacts.iter().find_map(|artifact| {
        let AgentArtifact::Text { text, .. } = artifact else {
            return None;
        };
        let candidate = semantic_json_object_candidate(text)?;
        serde_json::from_str(&candidate).ok()
    })
}

fn semantic_json_object_candidate(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed.to_string());
    }
    if let Some(after_fence) = trimmed.strip_prefix("```") {
        let after_language = after_fence
            .split_once('\n')
            .map(|(_, rest)| rest)
            .unwrap_or(after_fence);
        let fenced = after_language
            .rsplit_once("```")
            .map(|(before, _)| before)
            .unwrap_or(after_language)
            .trim();
        if fenced.starts_with('{') && fenced.ends_with('}') {
            return Some(fenced.to_string());
        }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then(|| trimmed[start..=end].to_string())
}

fn child_agent_input(request: &NormalizedRequest, task: &SubtaskSpec) -> String {
    let original = agent_visible_context(request);
    let contract = "Worker artifact contract:\n\
        - Return a compact intermediate artifact, not a chat transcript.\n\
        - You do not have tool access in this worker turn; do not request, invent, or simulate tool calls.\n\
        - Do not say you will inspect files, run commands, browse, or write code unless the evidence is already present in the root-visible context.\n\
        - Do not return spawn_plan JSON or orchestration schemas; planning is already complete.\n\
        - Focus only on this child objective. Prefer concrete findings, constraints, risks, and the next root-visible action if one is needed.\n\
        - Keep the artifact concise: at most 6 short bullets or one small JSON object. No hidden reasoning and no orchestration commentary.";
    if original.trim().is_empty() {
        format!("Child objective:\n{}\n\n{contract}", task.objective)
    } else {
        format!(
            "Child objective:\n{}\n\n{contract}\n\nOriginal request and root-visible context:\n{}",
            task.objective, original
        )
    }
}

fn synthesis_input(request: &NormalizedRequest, artifacts: &[AgentArtifact]) -> String {
    let artifact_text = synthesize_text(artifacts);
    synthesis_input_from_findings(request, &artifact_text)
}

fn root_tool_continuation_input(
    request: &NormalizedRequest,
    artifacts: &[AgentArtifact],
) -> String {
    let original = agent_visible_context(request);
    let artifact_text = synthesize_text(artifacts);
    let tool_policy = stateful_tool_policy_instruction(request)
        .map(|policy| format!("\n\nTool selection guardrails:\n{policy}"))
        .unwrap_or_default();
    format!(
        "User request and current root-visible tool context:\n{original}\n\nWorker-agent findings available to the root leader:\n{artifact_text}{tool_policy}\n\nUse these findings to continue the task. Before emitting a tool call, compare it against the root-visible tool context and do not rerun a command or equivalent validation whose successful result is already present. If the requested artifacts exist and a relevant validation succeeded, produce the final answer now. Emit another root-visible tool call only for a concrete missing artifact, failed validation, or unresolved error. When concrete environment work remains, use a direct execution or observation tool; planning/status-only tools should not delay file creation, command execution, retrieval, or validation. Batch safe related shell steps into one cohesive command/script when that reduces round trips without hiding failures."
    )
}

fn synthesis_input_from_findings(request: &NormalizedRequest, artifact_text: &str) -> String {
    let original = agent_visible_context(request);
    format!(
        "User request and root-visible context:\n{original}\n\nSummarized verified worker-agent findings:\n{artifact_text}\n\nReturn only a complete, natural final answer for the user. Use the summarized findings as guidance, but do not shorten the answer just because the findings are concise. Do not expose sub-agent state, internal tool calls, raw artifacts, or orchestration details. Preserve formatting exactly when the answer contains XML/HTML-like tags, Markdown, fenced code, lists, tables, or delimiter-separated blocks. Do not minify, collapse line breaks, remove spaces around headings, or merge tag-delimited sections."
    )
}

fn agent_visible_context(request: &NormalizedRequest) -> String {
    let mut sections = Vec::new();
    let original = flatten_text(request);
    if !original.trim().is_empty() {
        sections.push(original);
    }

    let tool_context = tool_context_text(request);
    if !tool_context.trim().is_empty() {
        sections.push(format!(
            "Root-visible tool context observed so far:\n{tool_context}"
        ));
    }

    sections.join("\n\n")
}

fn tool_context_text(request: &NormalizedRequest) -> String {
    let mut lines = Vec::new();
    let result_ids = request
        .tool_results
        .iter()
        .map(|result| result.tool_call_id.clone())
        .collect::<BTreeSet<_>>();

    for message in &request.messages {
        for part in &message.content {
            match part {
                NormalizedContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments_json,
                } => lines.push(format!(
                    "Tool call {} ({tool_name}) arguments:\n{}",
                    tool_call_id.as_ref(),
                    json_for_prompt(arguments_json)
                )),
                NormalizedContentPart::ToolResult { tool_call_id }
                    if !result_ids.contains(tool_call_id) =>
                {
                    lines.push(format!(
                        "Tool result {} is present in the conversation.",
                        tool_call_id.as_ref()
                    ));
                }
                _ => {}
            }
        }
    }

    for result in &request.tool_results {
        lines.push(format!(
            "Tool result {} ({:?}):\n{}",
            result.tool_call_id.as_ref(),
            result.status,
            json_for_prompt(&result.result_json)
        ));
    }

    lines.join("\n\n")
}

fn json_for_prompt(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
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

fn reasoning_summary_input(
    request: &NormalizedRequest,
    trace_events: &[KernelTraceEvent],
) -> String {
    let original = flatten_text(request);
    let worker_outputs = trace_events
        .iter()
        .filter_map(|event| {
            let KernelTraceEvent::AgentOutput {
                task_id,
                role,
                text_outputs,
                ..
            } = event
            else {
                return None;
            };
            if *role != AgentRole::Worker || text_outputs.is_empty() {
                return None;
            }
            Some(format!(
                "{} (Worker agent):\n{}",
                task_id.as_ref(),
                text_outputs.join("\n")
            ))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "User request:\n{original}\n\nWorker-agent outputs to summarize:\n{worker_outputs}\n\nReturn only a clean public reasoning summary. Use exactly one concise line per worker in this format:\n<task-id> (Worker agent): <summary>\nDo not include orchestration statistics, provider call counts, token counts, hidden reasoning, tool payloads, or the final answer."
    )
}

fn normalize_structured_text_layout(text: &str) -> String {
    if !looks_like_structured_text(text) {
        return text.to_string();
    }

    let with_tag_boundaries = normalize_xml_like_tag_boundaries(text);
    let with_heading_spacing = normalize_markdown_heading_spacing(&with_tag_boundaries);
    normalize_field_label_boundaries(&with_heading_spacing)
}

fn looks_like_structured_text(text: &str) -> bool {
    count_xml_like_tags(text) >= 2
        || text.contains("```")
        || text
            .lines()
            .any(|line| line.trim_start().starts_with('#') || line.trim_start().starts_with('|'))
}

fn count_xml_like_tags(text: &str) -> usize {
    let mut count = 0;
    let mut cursor = 0;
    while let Some(offset) = text[cursor..].find('<') {
        let start = cursor + offset;
        let Some(end) = xml_like_tag_end(text, start) else {
            cursor = start + 1;
            continue;
        };
        count += 1;
        cursor = end + 1;
    }
    count
}

fn normalize_xml_like_tag_boundaries(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 16);
    let mut cursor = 0;

    while cursor < text.len() {
        let Some(relative_start) = text[cursor..].find('<') else {
            output.push_str(&text[cursor..]);
            break;
        };
        let start = cursor + relative_start;
        output.push_str(&text[cursor..start]);

        let Some(end) = xml_like_tag_end(text, start) else {
            output.push('<');
            cursor = start + 1;
            continue;
        };

        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&text[start..=end]);
        if !output.ends_with('\n') {
            output.push('\n');
        }
        cursor = end + 1;
    }

    output.trim().to_string()
}

fn xml_like_tag_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start) != Some(&b'<') {
        return None;
    }
    let mut index = start + 1;
    if bytes.get(index) == Some(&b'/') {
        index += 1;
    }
    let first = *bytes.get(index)?;
    if !first.is_ascii_alphabetic() && first != b'_' {
        return None;
    }

    while let Some(byte) = bytes.get(index) {
        match *byte {
            b'>' => return Some(index),
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b':' | b'/' | b' ' | b'\t' => {
                index += 1
            }
            _ => return None,
        }
    }
    None
}

fn normalize_markdown_heading_spacing(text: &str) -> String {
    text.lines()
        .map(|line| {
            let trimmed = line.trim_start();
            let indent_len = line.len().saturating_sub(trimmed.len());
            let hashes = trimmed.chars().take_while(|&ch| ch == '#').count();
            if (1..=6).contains(&hashes) {
                let after_hashes = &trimmed[hashes..];
                if !after_hashes.is_empty() && !after_hashes.starts_with(char::is_whitespace) {
                    return format!(
                        "{}{} {}",
                        &line[..indent_len],
                        "#".repeat(hashes),
                        after_hashes
                    );
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_field_label_boundaries(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 16);
    let chars = text.chars().collect::<Vec<_>>();

    for (index, &ch) in chars.iter().enumerate() {
        if index > 0
            && !output.ends_with('\n')
            && output
                .chars()
                .rev()
                .take_while(|&existing| !matches!(existing, '\n' | '\r'))
                .any(|existing| matches!(existing, ':' | '：'))
            && is_field_label_boundary(&chars, index)
            && !matches!(chars[index - 1], ':' | '：' | '/' | '-' | '_' | '<' | '>')
        {
            output.push('\n');
        }
        output.push(ch);
    }

    output
}

fn is_field_label_boundary(chars: &[char], start: usize) -> bool {
    if !is_label_start(chars[start]) {
        return false;
    }

    let previous = chars[start - 1];
    if previous.is_whitespace() || matches!(previous, '\n' | '\r' | ':' | '：') {
        return false;
    }

    let mut end = start;
    while end < chars.len() && is_label_char(chars[end]) {
        end += 1;
    }

    if end >= chars.len() || !matches!(chars[end], ':' | '：') {
        return false;
    }

    let label_len = end.saturating_sub(start);
    if !(2..=32).contains(&label_len) {
        return false;
    }

    if chars[start].is_ascii_uppercase() {
        return previous.is_ascii_lowercase() || previous.is_ascii_digit() || is_cjk(previous);
    }

    if is_cjk(chars[start]) {
        return can_precede_dense_label(previous);
    }

    false
}

fn is_label_start(ch: char) -> bool {
    ch.is_ascii_uppercase() || is_cjk(ch)
}

fn is_label_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/') || is_cjk(ch)
}

fn can_precede_dense_label(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || is_cjk(ch)
        || matches!(
            ch,
            '%' | ')' | ']' | '}' | '）' | '】' | '」' | '』' | '》' | '〉'
        )
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x9fff | 0xf900..=0xfaff)
}

fn accumulate_usage(
    usage: &mut ProviderUsage,
    provider_call_count: &mut u32,
    response: &ProviderResponse,
) {
    *provider_call_count = provider_call_count.saturating_add(1);
    usage.input_tokens = usage
        .input_tokens
        .saturating_add(response.usage.input_tokens);
    usage.output_tokens = usage
        .output_tokens
        .saturating_add(response.usage.output_tokens);
}

fn trace_agent_input(task: &SubtaskSpec, input_text: String) -> KernelTraceEvent {
    KernelTraceEvent::AgentInput {
        task_id: task.task_id.clone(),
        role: task.role.clone(),
        objective: task.objective.clone(),
        input_text,
    }
}

fn trace_events_from_response(
    task: &SubtaskSpec,
    response: &ProviderResponse,
) -> Vec<KernelTraceEvent> {
    let mut events = Vec::new();
    for artifact in &response.artifacts {
        if let AgentArtifact::SpawnPlan { plan, .. } = artifact {
            events.push(KernelTraceEvent::SpawnPlan {
                task_id: task.task_id.clone(),
                reason: plan.reason.clone(),
                children: plan.children.clone(),
            });
        }
    }

    let text_outputs = response
        .artifacts
        .iter()
        .filter_map(|artifact| match artifact {
            AgentArtifact::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !text_outputs.is_empty() || !response.tool_calls.is_empty() {
        events.push(KernelTraceEvent::AgentOutput {
            task_id: task.task_id.clone(),
            role: task.role.clone(),
            text_outputs,
            tool_calls: response.tool_calls.clone(),
            usage: response.usage.clone(),
        });
    }

    events
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
            ..KernelPolicy::default()
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

    #[test]
    fn spawn_plan_budget_request_is_advisory_not_a_hard_failure() {
        let scope = IsolationKey::new("tenant", "request-a", "conversation");
        let root = TaskId::from("root");
        let store = ArtifactStore::new();
        let mut graph = TaskGraph::new(root.clone());
        let validator = SpawnValidator::new(KernelPolicy {
            limits: ExecutionLimits {
                max_total_tool_calls: 1,
                ..ExecutionLimits::default()
            },
            ..KernelPolicy::default()
        });
        let plan = SpawnPlan {
            parent_task_id: root.clone(),
            reason: "model requested more tool budget than policy permits".to_string(),
            children: vec![model_child("child-a", "do bounded child work")],
            expected_artifacts: vec![ArtifactKind::Text],
            budget_request: BudgetRequest {
                max_tokens: 64,
                max_tool_calls: 999,
            },
        };

        validator
            .validate_and_apply(&scope, &store, &mut graph, &plan)
            .unwrap();
        assert!(graph.tasks.contains_key(&TaskId::from("child-a")));
    }

    #[tokio::test]
    async fn model_orchestrator_decides_child_agent_division() {
        let request = text_request(
            "solve a compression task with independent planning, building, and verification",
        );
        let runner = KernelRunner::new(ModelPlanProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("inspect-format"))
        );
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("build-candidate"))
        );
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("verify-result"))
        );
        assert!(
            !output
                .task_graph
                .tasks
                .keys()
                .any(|task_id| task_id.as_ref().starts_with("deterministic-child-"))
        );
        assert_eq!(output.final_text, "model planned synthesis complete");
    }

    #[tokio::test]
    async fn tool_result_turn_returns_model_plan_findings_to_root_leader() {
        let mut request =
            text_request("create the required artifact from the observed command output");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-read"),
                tool_name: "exec_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "cat /app/decomp.c"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-read"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-read"),
            scope,
            result_json: serde_json::json!({"stdout": "decompressor source", "exit_code": 0}),
            result_sha256: "tool-result".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let provider = ToolResultModelPlanProvider::default();
        let seen = provider.seen.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(output.final_text, "leader pre-answer");
        let seen = seen.lock().unwrap();
        assert!(
            seen.iter()
                .any(|entry| entry.contains("planner saw tool result"))
        );
        assert!(
            !seen
                .iter()
                .any(|entry| entry.contains("synthesizer had no leader pre-answer"))
        );
        assert!(
            !output
                .task_graph
                .tasks
                .keys()
                .any(|task_id| task_id.as_ref().starts_with("deterministic-child-"))
        );
    }

    #[tokio::test]
    async fn tool_result_turn_orchestrates_worker_analysis_before_next_root_tool_call() {
        let mut request =
            text_request("inspect the command output and continue with the next command");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-read"),
                tool_name: "exec_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "cat /app/decomp.c"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-read"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-read"),
            scope,
            result_json: serde_json::json!({"stdout": "decompressor source", "exit_code": 0}),
            result_sha256: "tool-result".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let provider = ToolResultOrderProvider::default();
        let calls = provider.calls.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(!output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(
            output.tool_calls[0].tool_call_id,
            ToolCallId::from("call-next")
        );

        let calls = calls.lock().unwrap();
        assert!(calls[0].contains("orchestration plan"), "{calls:?}");
        assert!(
            calls
                .iter()
                .any(|call| call.contains("continue root-visible tool execution"))
        );
    }

    #[tokio::test]
    async fn initial_tool_turn_orchestrates_before_first_root_tool_call() {
        let mut request = text_request("create files with shell commands and verify the result");
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let provider = ToolResultOrderProvider::default();
        let calls = provider.calls.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(!output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].tool_name, "exec_command");

        let calls = calls.lock().unwrap();
        assert!(calls[0].contains("orchestration plan"), "{calls:?}");
        assert!(
            calls
                .iter()
                .any(|call| call.contains("continue root-visible tool execution")),
            "{calls:?}"
        );
        assert!(
            !calls[0].contains("continue root-visible tool execution"),
            "{calls:?}"
        );
    }

    #[test]
    fn stateful_tool_policy_blocks_invented_session_handles() {
        let mut request = text_request("continue the previous command");
        request.tools = vec![
            ToolDefinition {
                name: "exec_command".to_string(),
                description: Some("run a shell command".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]
                }),
            },
            ToolDefinition {
                name: "continue_active_operation".to_string(),
                description: Some("send input to an existing operation".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "integer"},
                        "chars": {"type": "string"}
                    },
                    "required": ["session_id", "chars"]
                }),
            },
        ];

        let instructions =
            system_instructions_with_policy(&request, &KernelPolicy::default()).join("\n");

        assert!(instructions.contains("Stateful tool policy"));
        assert!(instructions.contains("continue_active_operation"));
        assert!(instructions.contains("session_id"));
        assert!(instructions.contains("Do not invent numeric IDs"));
        assert!(instructions.contains("No state handle IDs have been observed"));
    }

    #[test]
    fn stateful_tool_policy_lists_observed_handles_from_tool_results() {
        let mut request = text_request("continue the running command");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "continue_active_operation".to_string(),
            description: Some("send input to an existing operation".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "chars": {"type": "string"}
                },
                "required": ["session_id", "chars"]
            }),
        }];
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-run"),
            scope,
            result_json: serde_json::json!({
                "status": "in_progress",
                "session_id": 42,
                "stdout": ""
            }),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];

        let instructions =
            system_instructions_with_policy(&request, &KernelPolicy::default()).join("\n");

        assert!(instructions.contains("Observed state handle IDs"));
        assert!(instructions.contains("session_id=42"));
    }

    #[test]
    fn root_tool_continuation_prompt_stops_redundant_tool_loops() {
        let mut request = text_request("create and verify an artifact");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "run_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-create"),
                tool_name: "run_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "create artifact && verify artifact"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-create"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-create"),
            scope,
            result_json: serde_json::json!({
                "exit_code": 0,
                "stdout": "artifact created\nvalidation successful\n"
            }),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let artifacts = vec![AgentArtifact::Text {
            id: ArtifactId::from("finding"),
            scope: request.isolation_key(),
            text: "Worker confirms the artifact exists and validation passed.".to_string(),
        }];

        let prompt = root_tool_continuation_input(&request, &artifacts);

        assert!(prompt.contains("do not rerun a command or equivalent validation"));
        assert!(prompt.contains("requested artifacts exist and a relevant validation succeeded"));
        assert!(prompt.contains("produce the final answer now"));
        assert!(prompt.contains("failed validation"));
        assert!(prompt.contains("planning/status-only tools should not delay"));
        assert!(prompt.contains("Batch safe related shell steps into one cohesive command/script"));
    }

    #[test]
    fn child_agent_contract_rejects_spawn_plan_output() {
        let request = text_request("verify a generated artifact");
        let task = model_child("verify-script", "verify generated script");

        let prompt = child_agent_input(&request, &task);

        assert!(prompt.contains("Do not return spawn_plan JSON"));
    }

    #[tokio::test]
    async fn invalid_model_plan_is_rejected_without_root_fallback() {
        let request = text_request("complex request with an invalid planning response");
        let runner = KernelRunner::new(InvalidPlannerProvider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(error, KernelError::ProviderRejected(message)
            if message.contains("refusing root-only fallback")));
    }

    #[tokio::test]
    async fn planner_repair_expands_under_target_model_plan_without_template_children() {
        let request = text_request("compare implementation options across many independent risks");
        let provider = PlannerRepairProvider::default();
        let planner_inputs = provider.planner_inputs.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert!(
            !output
                .task_graph
                .tasks
                .keys()
                .any(|task_id| task_id.as_ref().starts_with("deterministic-child-"))
        );
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("model-expanded-15"))
        );

        let planner_inputs = planner_inputs.lock().unwrap();
        assert_eq!(planner_inputs.len(), 2);
        assert!(planner_inputs[1].contains("Previous orchestration attempt"));
        assert!(planner_inputs[1].contains("target_child_agents=16"));
    }

    #[tokio::test]
    async fn internal_orchestration_preserves_large_output_caps_and_sets_planner_contract() {
        let mut request = text_request("spawn with tiny client cap");
        request.reasoning_effort = ReasoningEffort::Low;
        request.provider_options = serde_json::json!({
            "max_tokens": 8,
            "max_new_tokens": 65536,
            "temperature": 0.2,
            "stop": ["client-facing-stop"],
            "stopping_strings": ["client-facing-stop"]
        });
        let provider = ProviderOptionFloorProvider::default();
        let seen = provider.seen.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert_eq!(output.encrypted_subagent_state.len(), 4);
        let seen = seen.lock().unwrap();
        assert!(seen.iter().any(|entry| {
            entry.role == AgentRole::Leader
                && entry.objective.contains("orchestration plan")
                && entry.options["max_tokens"] == 2048
                && entry.options["max_new_tokens"] == 65536
                && entry.options["temperature"] == 0
                && entry.options["response_format"]["type"] == "json_object"
                && entry.options.get("stop").is_none()
                && entry.options.get("stopping_strings").is_none()
        }));
        assert!(seen.iter().any(|entry| {
            entry.role == AgentRole::Worker
                && entry.options["max_tokens"] == 2048
                && entry.options["max_new_tokens"] == 65536
                && entry.options["temperature"] == 0.2
                && entry.options.get("stop").is_none()
                && entry.options.get("stopping_strings").is_none()
        }));
        assert!(seen.iter().any(|entry| {
            entry.role == AgentRole::Synthesizer
                && entry.options["max_tokens"] == 8
                && entry.options["max_new_tokens"] == 65536
                && entry.options["temperature"] == 0.2
                && entry.options["stop"] == serde_json::json!(["client-facing-stop"])
                && entry.options["stopping_strings"] == serde_json::json!(["client-facing-stop"])
        }));
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
            public_reasoning_enabled: false,
            provider_options: serde_json::json!({}),
            metadata: serde_json::json!({}),
        };

        let runner = KernelRunner::new(MockProvider, KernelPolicy::default());
        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert!(output.final_text.contains("clear, usable answer"));
        assert!(!output.final_text.contains("child completed"));
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert!(
            output
                .task_graph
                .tasks
                .contains_key(&TaskId::from("child-00"))
        );
    }

    #[tokio::test]
    async fn only_root_tool_calls_are_public_and_child_state_is_encrypted() {
        let request = text_request("spawn child tool");
        let runner = KernelRunner::new(ChildToolProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.tool_calls.is_empty());
        assert_eq!(output.final_text, "Final answer from main synthesizer.");
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        for sealed in &output.encrypted_subagent_state {
            assert!(sealed.task_id.as_ref().starts_with("child-tool-"));
            assert!(!sealed.ciphertext.contains("child-secret-output"));
            assert!(!sealed.ciphertext.contains("private_lookup"));
        }
    }

    #[tokio::test]
    async fn child_tasks_run_in_parallel() {
        let mut request = text_request("spawn slow children");
        request.reasoning_effort = ReasoningEffort::Low;
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
    async fn medium_default_generates_sixteen_child_agents_with_four_in_flight() {
        let request = text_request("spawn six parallel probe children");
        let provider = ParallelProbeProvider::new(16);
        let max_in_flight = provider.max_in_flight.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(max_in_flight.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn metadata_max_parallel_agents_limits_in_flight_requests_not_child_agent_count() {
        let mut request = text_request("spawn six parallel probe children");
        request.metadata = serde_json::json!({
            "agent": {"max_parallel_agents": 2}
        });
        let provider = ParallelProbeProvider::new(16);
        let max_in_flight = provider.max_in_flight.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(max_in_flight.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn root_only_request_does_not_call_synthesizer() {
        let mut request = text_request("simple answer");
        request.reasoning_effort = ReasoningEffort::None;
        let provider = CountingProvider::default();
        let calls = provider.calls.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(output.final_text, "root final answer");
    }

    #[tokio::test]
    async fn model_orchestrator_rejects_missing_spawn_plan_without_template_fallback() {
        let request = text_request("ordinary request that still needs bounded orchestration");
        let provider = NoSpawnParallelProvider::default();
        let max_in_flight = provider.max_in_flight.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(error, KernelError::ProviderRejected(message)
            if message.contains("refusing root-only fallback")));
        assert_eq!(max_in_flight.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn model_planned_child_agents_receive_original_user_request() {
        let request = text_request("remember this exact phrase: alpha-beta");
        let provider = RecordingChildInputProvider::default();
        let child_inputs = provider.child_inputs.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        let child_inputs = child_inputs.lock().unwrap();
        assert_eq!(child_inputs.len(), 16);
        assert!(child_inputs.iter().all(|input| {
            input.contains("Child objective:")
                && input.contains("Original request and root-visible context:")
                && input.contains("Worker artifact contract:")
                && input.contains("do not request, invent, or simulate tool calls")
                && input.contains("at most 6 short bullets")
                && input.contains("remember this exact phrase: alpha-beta")
        }));
    }

    #[tokio::test]
    async fn tool_context_answer_turn_uses_model_planned_children() {
        let mut request = text_request("summarize the command output");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-1"),
                tool_name: "exec_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "printf needle"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-1"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope,
            result_json: serde_json::json!({"stdout": "needle found", "exit_code": 0}),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let provider = RecordingChildInputProvider::default();
        let child_inputs = provider.child_inputs.clone();
        let seen_tools = provider.seen_tools.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(output.provider_call_count, 18);
        assert_eq!(output.final_text, "recorded root output");

        let child_inputs = child_inputs.lock().unwrap();
        assert_eq!(child_inputs.len(), 16);
        assert!(child_inputs.iter().all(|input| {
            input.contains("Root-visible tool context observed so far:")
                && input.contains("Worker artifact contract:")
                && input.contains("do not request, invent, or simulate tool calls")
                && input.contains("exec_command")
                && input.contains("printf needle")
                && input.contains("needle found")
        }));

        let seen_tools = seen_tools.lock().unwrap();
        assert!(seen_tools.iter().any(|entry| entry == "Leader:1"));
        assert_eq!(
            seen_tools
                .iter()
                .filter(|entry| entry.as_str() == "Worker:0")
                .count(),
            16
        );
        assert_eq!(
            seen_tools
                .iter()
                .filter(|entry| entry.as_str() == "Leader:1")
                .count(),
            1
        );
        assert!(!seen_tools.iter().any(|entry| entry == "Synthesizer:0"));
        assert!(!seen_tools.iter().any(|entry| entry == "Worker:1"));
    }

    #[tokio::test]
    async fn tool_result_turn_can_continue_root_tool_loop() {
        let mut request = text_request("continue after observing the command output");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-1"),
                tool_name: "exec_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "python3 --version"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-1"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope,
            result_json: serde_json::json!({"stderr": "python3: command not found", "exit_code": 127}),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let runner = KernelRunner::new(ToolContinuationProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(!output.verification.passed);
        assert_eq!(output.provider_call_count, 18);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert!(output.final_text.is_empty());
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].tool_name, "exec_command");
        assert_eq!(output.tool_calls[0].arguments_json["cmd"], "ls -la /app");
    }

    #[tokio::test]
    async fn worker_findings_return_to_root_for_next_tool_call() {
        let mut request = text_request("fix the failed command and continue");
        let scope = request.isolation_key();
        request.tools = vec![ToolDefinition {
            name: "exec_command".to_string(),
            description: Some("run a shell command".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages.push(NormalizedMessage {
            role: MessageRole::Assistant,
            content: vec![NormalizedContentPart::ToolCall {
                tool_call_id: ToolCallId::from("call-bad"),
                tool_name: "exec_command".to_string(),
                arguments_json: serde_json::json!({"cmd": "opensslget -subject"}),
            }],
        });
        request.messages.push(NormalizedMessage {
            role: MessageRole::Tool,
            content: vec![NormalizedContentPart::ToolResult {
                tool_call_id: ToolCallId::from("call-bad"),
            }],
        });
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-bad"),
            scope,
            result_json: serde_json::json!({"stderr": "opensslget: command not found", "exit_code": 127}),
            result_sha256: "result-hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let runner = KernelRunner::new(
            WorkerFindingToolContinuationProvider,
            KernelPolicy::default(),
        );

        let output = runner.run(request).await.unwrap();

        assert!(!output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 16);
        assert_eq!(output.provider_call_count, 18);
        assert!(output.final_text.is_empty());
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(
            output.tool_calls[0].arguments_json["cmd"],
            "openssl x509 -subject -noout -in /app/ssl/server.crt"
        );
    }

    #[tokio::test]
    async fn reasoning_effort_none_does_not_create_deterministic_children() {
        let mut request = text_request("direct root only");
        request.reasoning_effort = ReasoningEffort::None;
        let runner = KernelRunner::new(NoSpawnParallelProvider::default(), KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 0);
        assert_eq!(output.provider_call_count, 1);
        assert_eq!(output.final_text, "root final answer");
    }

    #[tokio::test]
    async fn kernel_enforces_agent_timeout() {
        let mut policy = KernelPolicy::default();
        policy.limits.request_timeout_ms = 100;
        policy.limits.agent_timeout_ms = 5;
        let runner = KernelRunner::new(DelayedProvider { delay_ms: 25 }, policy);

        let error = runner.run(text_request("timeout agent")).await.unwrap_err();

        assert!(matches!(
            error,
            KernelError::AgentTimeout { timeout_ms: 5, .. }
        ));
    }

    #[tokio::test]
    async fn kernel_enforces_request_timeout() {
        let mut policy = KernelPolicy::default();
        policy.limits.request_timeout_ms = 5;
        policy.limits.agent_timeout_ms = 100;
        let runner = KernelRunner::new(DelayedProvider { delay_ms: 25 }, policy);

        let error = runner
            .run(text_request("timeout request"))
            .await
            .unwrap_err();

        assert_eq!(error, KernelError::RequestTimeout { timeout_ms: 5 });
    }

    #[derive(Default)]
    struct SemanticRepairProvider {
        verifier_calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ModelProvider for SemanticRepairProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            if request.task.role == AgentRole::Verifier {
                let call = self
                    .verifier_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let passed = call > 0;
                let issues = if passed {
                    Vec::new()
                } else {
                    vec![serde_json::json!({
                        "code": "missing_evidence",
                        "message": "candidate must explicitly cite the worker evidence"
                    })]
                };
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("verdict-{call}")),
                        scope: request.scope,
                        text: serde_json::json!({
                            "passed": passed,
                            "issues": issues,
                            "covered_artifact_ids": request
                                .artifacts
                                .iter()
                                .map(|artifact| artifact.id().as_ref().to_string())
                                .collect::<Vec<_>>()
                        })
                        .to_string(),
                    }],
                    tool_calls: Vec::new(),
                    usage: ProviderUsage::default(),
                });
            }
            if request.task.role == AgentRole::Synthesizer
                && request.task.objective.contains("repair the candidate")
            {
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("semantic-repair-output"),
                        scope: request.scope,
                        text: "Repaired answer grounded in worker evidence.".to_string(),
                    }],
                    tool_calls: Vec::new(),
                    usage: ProviderUsage::default(),
                });
            }
            MockProvider.invoke(request).await
        }
    }

    #[tokio::test]
    async fn semantic_verifier_checks_artifact_coverage_after_synthesis() {
        let mut policy = KernelPolicy::default();
        policy.semantic_verification.enabled = true;
        policy.semantic_verification.max_repair_attempts = 2;
        let runner = KernelRunner::new(MockProvider, policy);
        let output = runner
            .run(text_request("spawn workers and produce a verified answer"))
            .await
            .unwrap();

        assert!(output.verification.passed);
        assert!(!output.verification.artifact_coverage.is_empty());
        assert!(
            output
                .verification
                .artifact_coverage
                .iter()
                .all(|coverage| coverage.covered)
        );
        assert!(output.trace_events.iter().any(|event| matches!(
            event,
            KernelTraceEvent::AgentInput {
                role: AgentRole::Verifier,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn semantic_verifier_runs_bounded_repair_and_rechecks() {
        let mut policy = KernelPolicy::default();
        policy.semantic_verification.enabled = true;
        policy.semantic_verification.max_repair_attempts = 1;
        let provider = std::sync::Arc::new(SemanticRepairProvider::default());
        let runner = KernelRunner::new(provider.clone(), policy);
        let output = runner
            .run(text_request("spawn workers and repair a grounded answer"))
            .await
            .unwrap();

        assert!(output.verification.passed);
        assert_eq!(
            provider
                .verifier_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        assert_eq!(
            output.final_text,
            "Repaired answer grounded in worker evidence."
        );
    }

    #[tokio::test]
    async fn verification_records_token_usage_without_failing_budget() {
        let mut request = text_request("oversized answer");
        request.reasoning_effort = ReasoningEffort::None;
        let runner = KernelRunner::new(BudgetOverflowProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.verification.budget_summary.token_budget, 16_384);
        assert_eq!(output.verification.budget_summary.token_used, 32_701);
        assert!(output.verification.issues.is_empty());
    }

    #[tokio::test]
    async fn tool_call_budget_summary_counts_provider_calls() {
        let mut request = text_request("please use a tool");
        request.tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: Some("public client tool".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
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
        assert_eq!(seen.first().map(String::as_str), Some("Leader:0"));
        assert_eq!(
            seen.iter()
                .filter(|entry| entry.as_str() == "Leader:1")
                .count(),
            1
        );
        assert_eq!(
            seen.iter()
                .filter(|entry| entry.as_str() == "Worker:0")
                .count(),
            16
        );
        assert!(!seen.iter().any(|entry| entry == "Worker:1"));
        assert!(!seen.iter().any(|entry| entry == "Synthesizer:1"));
    }

    #[tokio::test]
    async fn kernel_aggregates_provider_usage_across_agents() {
        let request = text_request("spawn usage children");
        let runner = KernelRunner::new(UsageFanoutProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert_eq!(output.provider_call_count, 18);
        assert_eq!(output.usage.input_tokens, 94);
        assert_eq!(output.usage.output_tokens, 127);
        assert_eq!(output.verification.budget_summary.token_used, 221);
    }

    #[tokio::test]
    async fn kernel_trace_captures_spawn_child_and_synthesizer_outputs() {
        let request = text_request("spawn usage children");
        let runner = KernelRunner::new(UsageFanoutProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(
            output
                .trace_events
                .iter()
                .any(|event| matches!(event, KernelTraceEvent::SpawnPlan { .. }))
        );
        assert!(output.trace_events.iter().any(|event| matches!(
            event,
            KernelTraceEvent::AgentOutput {
                task_id,
                text_outputs,
                ..
            } if task_id.as_ref() == "usage-child-00"
                && text_outputs.iter().any(|text| text.contains("work"))
        )));
        assert!(output.trace_events.iter().any(|event| matches!(
            event,
            KernelTraceEvent::AgentOutput {
                task_id,
                text_outputs,
                ..
            } if task_id.as_ref() == "synthesizer"
                && text_outputs.iter().any(|text| text.contains("usage final answer"))
        )));
    }

    #[tokio::test]
    async fn kernel_uses_summary_agent_between_workers_and_final_synthesis() {
        let provider = SummaryPipelineProvider::default();
        let calls = provider.calls.clone();
        let runner = KernelRunner::new(provider, KernelPolicy::default());
        let mut request = text_request("compare the implementation options");
        request.public_reasoning_enabled = true;
        let output = runner.run(request).await.unwrap();

        assert_eq!(
            output.final_text,
            "final answer from original worker findings"
        );
        let calls = calls.lock().unwrap();
        let summary_index = calls
            .iter()
            .position(|call| call.task_id == "reasoning-summary")
            .expect("summary agent was not called");
        let synthesis_index = calls
            .iter()
            .position(|call| call.task_id == "synthesizer")
            .expect("synthesizer was not called");
        assert!(summary_index < synthesis_index);

        let synth_input = calls
            .iter()
            .find(|call| call.task_id == "synthesizer")
            .unwrap()
            .input_text
            .as_str();
        assert!(synth_input.contains("RAW-WORKER-DETAIL"));
        assert!(!synth_input.contains("summarized child 1"));
    }

    #[tokio::test]
    async fn reasoning_effort_low_limits_request_to_four_agents() {
        let mut request = text_request("spawn five children");
        request.reasoning_effort = ReasoningEffort::Low;
        request.tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: Some("public client tool".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let runner = KernelRunner::new(FiveChildProvider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(
            error,
            KernelError::AgentLimitExceeded {
                requested: 5,
                max: 4
            }
        ));
    }

    #[tokio::test]
    async fn provider_tool_calls_cannot_exceed_total_budget() {
        let mut request = text_request("too many tools");
        request.reasoning_effort = ReasoningEffort::None;
        let runner = KernelRunner::new(ManyToolCallsProvider, KernelPolicy::default());

        let error = runner.run(request).await.unwrap_err();

        assert!(matches!(error, KernelError::BudgetExceeded));
    }

    #[tokio::test]
    async fn reasoning_effort_high_allows_more_than_four_agents() {
        let mut request = text_request("spawn five children");
        request.reasoning_effort = ReasoningEffort::High;
        let runner = KernelRunner::new(ParallelProbeProvider::new(32), KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert!(output.verification.passed);
        assert_eq!(output.encrypted_subagent_state.len(), 32);
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
        assert!(instructions.contains("max_agents_per_request=32"));
        assert!(instructions.contains("target_parallel_agents=32"));
    }

    #[tokio::test]
    async fn higher_reasoning_effort_improves_model_planned_eval_coverage() {
        let low = run_effort_coverage_eval(ReasoningEffort::Low).await;
        let high = run_effort_coverage_eval(ReasoningEffort::High).await;
        let xhigh = run_effort_coverage_eval(ReasoningEffort::XHigh).await;

        assert_eq!(low.coverage_score, 4);
        assert_eq!(high.coverage_score, 32);
        assert_eq!(xhigh.coverage_score, 64);
        assert!(high.coverage_score > low.coverage_score);
        assert!(xhigh.coverage_score >= high.coverage_score);
        assert_eq!(low.child_agents, 4);
        assert_eq!(high.child_agents, 32);
    }

    #[test]
    fn synthesis_prompt_preserves_structured_markup_formatting() {
        let request = text_request("<document>\n### Heading\n</document>");
        let prompt = synthesis_input(&request, &[]);

        assert!(prompt.contains("Preserve formatting exactly"));
        assert!(prompt.contains("Do not minify"));
        assert!(prompt.contains("XML/HTML-like tags"));
        assert!(prompt.contains("Markdown"));
    }

    #[tokio::test]
    async fn kernel_formats_dense_structured_synthesis_output_with_generic_rules() {
        let request = text_request("return a generic structured document");
        let runner = KernelRunner::new(DenseStructuredProvider, KernelPolicy::default());

        let output = runner.run(request).await.unwrap();

        assert_eq!(output.provider_call_count, 18);
        assert!(
            output
                .final_text
                .contains("<doc>\n<section>\n### Heading\n")
        );
        assert!(
            output
                .final_text
                .contains("FieldOne: alpha\nFieldTwo: beta"),
            "{}",
            output.final_text
        );
        assert!(output.final_text.contains("</section>\n</doc>"));
        assert!(!output.final_text.contains("<doc><section>###Heading"));
        assert!(!output.final_text.contains("alphaFieldTwo"));
    }

    #[test]
    fn structured_layout_rules_split_dense_cjk_labels_generically() {
        let normalized = normalize_structured_text_layout("<doc>項目一：甲項目二：乙</doc>");

        assert_eq!(normalized, "<doc>\n項目一：甲\n項目二：乙\n</doc>");
    }

    struct EffortCoverageEval {
        coverage_score: usize,
        child_agents: usize,
    }

    async fn run_effort_coverage_eval(effort: ReasoningEffort) -> EffortCoverageEval {
        let mut request = text_request("deep compare across eight independent dimensions");
        request.reasoning_effort = effort;
        if matches!(
            request.reasoning_effort,
            ReasoningEffort::High | ReasoningEffort::XHigh
        ) {
            request.metadata = serde_json::json!({
                "agent": {"max_parallel_agents": 8}
            });
        }
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
            public_reasoning_enabled: false,
            provider_options: serde_json::json!({}),
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
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    let target = target_child_agents(&request);
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-child-tool"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: request.task.task_id,
                                reason: "delegate private lookup".to_string(),
                                children: (0..target)
                                    .map(|index| SubtaskSpec {
                                        task_id: TaskId::from(format!("child-tool-{index:02}")),
                                        parent_task_id: Some(TaskId::from("root")),
                                        spawn_depth: 1,
                                        role: AgentRole::Worker,
                                        objective: format!("private child tool work {index}"),
                                        input_artifact_refs: vec![],
                                        expected_outputs: vec![ArtifactKind::Text],
                                        allowed_capabilities: CapabilitySet::from([
                                            Capability::Text,
                                            Capability::ToolCall,
                                        ]),
                                        limits: AgentLimits::default(),
                                    })
                                    .collect(),
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 128,
                                    max_tool_calls: 1,
                                },
                            },
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
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
                AgentRole::ReasoningSummarizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("reasoning-summary"),
                        scope: request.scope,
                        text: request.input_text,
                    }],
                    tool_calls: vec![],
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

    #[derive(Clone)]
    struct ModelPlanProvider;

    #[async_trait::async_trait]
    impl ModelProvider for ModelPlanProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    let target = target_child_agents(&request);
                    let mut children = vec![
                        model_child(
                            "inspect-format",
                            "inspect the decompressor protocol and derive constraints",
                        ),
                        model_child(
                            "build-candidate",
                            "construct the candidate compressed artifact",
                        ),
                        model_child(
                            "verify-result",
                            "verify the generated artifact against the decompressor",
                        ),
                    ];
                    children.extend(model_children(
                        "implementation-risk",
                        target.saturating_sub(children.len()),
                        "model-selected supporting implementation risk slice",
                    ));
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-model-selected"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: TaskId::from("root"),
                                reason: "model selected distinct implementation phases".to_string(),
                                children,
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 768,
                                    max_tool_calls: 0,
                                },
                            },
                        }],
                        tool_calls: vec![],
                        usage: ProviderUsage {
                            input_tokens: 7,
                            output_tokens: 11,
                        },
                    })
                }
                AgentRole::Worker => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: format!(
                            "worker {} completed {}",
                            request.task.task_id.as_ref(),
                            request.task.objective
                        ),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("model-plan-final"),
                        scope: request.scope,
                        text: "model planned synthesis complete".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Leader | AgentRole::ReasoningSummarizer | AgentRole::Verifier => {
                    Ok(ProviderResponse::default())
                }
            }
        }
    }

    #[derive(Clone, Default)]
    struct ToolResultModelPlanProvider {
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for ToolResultModelPlanProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    if request.input_text.contains("decompressor source") {
                        self.seen
                            .lock()
                            .unwrap()
                            .push("planner saw tool result".to_string());
                    }
                    let target = target_child_agents(&request);
                    let mut children = vec![
                        model_child("derive-encoder", "derive an encoder from decomp.c"),
                        model_child(
                            "verify-by-command",
                            "verify cat data.comp | /app/decomp against data.txt",
                        ),
                    ];
                    children.extend(model_children(
                        "tool-evidence-slice",
                        target.saturating_sub(children.len()),
                        "analyze observed tool evidence slice",
                    ));
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-tool-result"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: TaskId::from("root"),
                                reason: "model used tool output to choose concrete workers"
                                    .to_string(),
                                children,
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 512,
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
                        text: format!("worker result: {}", request.task.objective),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => {
                    if !request.input_text.contains("leader pre-answer") {
                        self.seen
                            .lock()
                            .unwrap()
                            .push("synthesizer had no leader pre-answer".to_string());
                    }
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("tool-result-final"),
                            scope: request.scope,
                            text: "final artifact instructions".to_string(),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Leader | AgentRole::ReasoningSummarizer | AgentRole::Verifier => {
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("leader-pre-answer"),
                            scope: request.scope,
                            text: "leader pre-answer".to_string(),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
            }
        }
    }

    #[derive(Clone, Default)]
    struct ToolResultOrderProvider {
        calls: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for ToolResultOrderProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.calls.lock().unwrap().push(format!(
                "{:?}:{}",
                request.task.role, request.task.objective
            ));

            if request.task.role == AgentRole::Leader
                && request
                    .task
                    .objective
                    .contains("continue root-visible tool execution")
            {
                return Ok(ProviderResponse {
                    artifacts: vec![],
                    tool_calls: vec![ToolCallRecord {
                        tool_call_id: ToolCallId::from("call-next"),
                        scope: request.scope.clone(),
                        task_id: request.task.task_id,
                        agent_id: AgentId::from("agent-root"),
                        tool_name: "exec_command".to_string(),
                        arguments_sha256: "hash".to_string(),
                        arguments_json: serde_json::json!({"cmd": "python3 build_encoder.py"}),
                        status: ToolCallStatus::Pending,
                        created_at_ms: 0,
                        resolved_at_ms: None,
                    }],
                    usage: Default::default(),
                });
            }

            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                let mut children = vec![
                    model_child("decode-format", "derive encoding constraints from decomp.c"),
                    model_child(
                        "plan-command",
                        "identify the next command to build data.comp",
                    ),
                ];
                children.extend(model_children(
                    "next-tool-analysis",
                    target.saturating_sub(children.len()),
                    "analyze one independent aspect before next root-visible command",
                ));
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-tool-result-order"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason:
                                "analyze tool result before selecting next root-visible command"
                                    .to_string(),
                            children,
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }

            let text = match request.task.role {
                AgentRole::Worker => format!("worker finding: {}", request.task.objective),
                AgentRole::Synthesizer => "unexpected synthesis".to_string(),
                AgentRole::ReasoningSummarizer => "summary".to_string(),
                AgentRole::Leader | AgentRole::Verifier => {
                    "unexpected root direct answer".to_string()
                }
            };

            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text,
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone)]
    struct InvalidPlannerProvider;

    #[async_trait::async_trait]
    impl ModelProvider for InvalidPlannerProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("invalid-plan"),
                            scope: request.scope,
                            text: "I should probably split this, but I am not returning JSON."
                                .to_string(),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("root-final"),
                        scope: request.scope,
                        text: "root final after invalid plan".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Worker
                | AgentRole::ReasoningSummarizer
                | AgentRole::Synthesizer
                | AgentRole::Verifier => Ok(ProviderResponse::default()),
            }
        }
    }

    #[derive(Clone, Default)]
    struct PlannerRepairProvider {
        planner_inputs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for PlannerRepairProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    self.planner_inputs
                        .lock()
                        .unwrap()
                        .push(request.input_text.clone());
                    let repair = request
                        .input_text
                        .contains("Previous orchestration attempt");
                    let child_count = if repair { 16 } else { 2 };
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from(if repair {
                                "spawn-plan-repaired"
                            } else {
                                "spawn-plan-under-target"
                            }),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: TaskId::from("root"),
                                reason: if repair {
                                    "model repaired coverage to match the configured tier"
                                } else {
                                    "model initially under-covered the task"
                                }
                                .to_string(),
                                children: (0..child_count)
                                    .map(|index| {
                                        model_child(
                                            &format!("model-expanded-{index:02}"),
                                            &format!("model-selected risk slice {index}"),
                                        )
                                    })
                                    .collect(),
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 4096,
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
                        text: format!("worker completed {}", request.task.objective),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("repair-final"),
                        scope: request.scope,
                        text: "repair synthesis complete".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Leader | AgentRole::ReasoningSummarizer | AgentRole::Verifier => {
                    Ok(ProviderResponse::default())
                }
            }
        }
    }

    #[derive(Clone, Debug)]
    struct SeenProviderOptions {
        role: AgentRole,
        objective: String,
        options: serde_json::Value,
    }

    #[derive(Debug)]
    struct DelayedProvider {
        delay_ms: u64,
    }

    #[async_trait::async_trait]
    impl ModelProvider for DelayedProvider {
        async fn invoke(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(ProviderResponse::default())
        }
    }

    #[derive(Clone, Default)]
    struct ProviderOptionFloorProvider {
        seen: std::sync::Arc<std::sync::Mutex<Vec<SeenProviderOptions>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for ProviderOptionFloorProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.seen.lock().unwrap().push(SeenProviderOptions {
                role: request.task.role.clone(),
                objective: request.task.objective.clone(),
                options: request.provider_options.clone(),
            });

            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-option-floor"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason: "verify internal output cap floor".to_string(),
                            children: model_children(
                                "option-floor-child",
                                target,
                                "inspect option floor",
                            ),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }

            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: "option floor response".to_string(),
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone, Default)]
    struct SummaryPipelineProvider {
        calls: std::sync::Arc<std::sync::Mutex<Vec<SummaryPipelineCall>>>,
    }

    #[derive(Clone, Debug)]
    struct SummaryPipelineCall {
        task_id: String,
        input_text: String,
    }

    #[async_trait::async_trait]
    impl ModelProvider for SummaryPipelineProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.calls.lock().unwrap().push(SummaryPipelineCall {
                task_id: request.task.task_id.as_ref().to_string(),
                input_text: request.input_text.clone(),
            });

            let text = match request.task.task_id.as_ref() {
                "root" if request.task.objective.contains("orchestration plan") => {
                    let target = target_child_agents(&request);
                    return Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-summary-pipeline"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: TaskId::from("root"),
                                reason: "model selected comparison workers".to_string(),
                                children: model_children("summary-child", target, "compare option"),
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 512,
                                    max_tool_calls: 0,
                                },
                            },
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    });
                }
                "reasoning-summary" => "summary-child-a (Worker agent): summarized child 1\n\
                     summary-child-b (Worker agent): summarized child 2"
                    .to_string(),
                "synthesizer" => {
                    assert!(
                        request.input_text.contains("RAW-WORKER-DETAIL"),
                        "synthesizer input should preserve raw worker findings: {}",
                        request.input_text
                    );
                    assert!(
                        !request.input_text.contains("summarized child 1"),
                        "synthesizer input should not consume public summary text: {}",
                        request.input_text
                    );
                    "final answer from original worker findings".to_string()
                }
                task_id if request.task.role == AgentRole::Worker => {
                    format!("RAW-WORKER-DETAIL from {task_id}: {}", "x".repeat(512))
                }
                _ => String::new(),
            };

            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text,
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone, Default)]
    struct NoSpawnParallelProvider {
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for NoSpawnParallelProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Worker => {
                    let current = self
                        .in_flight
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        + 1;
                    self.max_in_flight
                        .fetch_max(current, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    self.in_flight
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                            scope: request.scope,
                            text: format!("worker result: {}", request.task.objective),
                        }],
                        tool_calls: vec![],
                        usage: ProviderUsage {
                            input_tokens: 11,
                            output_tokens: 13,
                        },
                    })
                }
                AgentRole::ReasoningSummarizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("reasoning-summary"),
                        scope: request.scope,
                        text: request.input_text,
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("deterministic-final"),
                        scope: request.scope,
                        text: "deterministic synthesis complete".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 17,
                        output_tokens: 19,
                    },
                }),
                AgentRole::Leader | AgentRole::Verifier => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: "root final answer".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 3,
                        output_tokens: 5,
                    },
                }),
            }
        }
    }

    #[derive(Clone, Default)]
    struct RecordingChildInputProvider {
        child_inputs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        seen_tools: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for RecordingChildInputProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            self.seen_tools.lock().unwrap().push(format!(
                "{:?}:{}",
                request.task.role,
                request.tools.len()
            ));
            if matches!(request.task.role, AgentRole::Worker) {
                self.child_inputs
                    .lock()
                    .unwrap()
                    .push(request.input_text.clone());
            }

            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-recording"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason: "record model-selected child inputs".to_string(),
                            children: model_children(
                                "record-child",
                                target,
                                "record independent view",
                            ),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }

            let text = match request.task.role {
                AgentRole::ReasoningSummarizer => "recorded child summary".to_string(),
                AgentRole::Synthesizer => "recorded synthesis".to_string(),
                AgentRole::Worker => "recorded child output".to_string(),
                AgentRole::Leader | AgentRole::Verifier => "recorded root output".to_string(),
            };

            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text,
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone)]
    struct ToolContinuationProvider;

    #[async_trait::async_trait]
    impl ModelProvider for ToolContinuationProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            assert!(
                request
                    .system_instructions
                    .iter()
                    .any(|instruction| instruction.contains("Tool execution policy")),
                "tool-capable requests must carry root tool-loop guidance"
            );
            if request.task.objective.contains("orchestration plan") {
                assert_eq!(request.task.role, AgentRole::Leader);
                let target = target_child_agents(&request);
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-tool-continuation"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason:
                                "mock provider analyzes tool turn before returning to root leader"
                                    .to_string(),
                            children: model_children(
                                "tool-continuation-child",
                                target,
                                "analyze command failure before root continuation",
                            ),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 0,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }
            if request.task.role == AgentRole::Worker {
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: format!("worker finding: {}", request.task.objective),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }
            assert_eq!(request.task.role, AgentRole::Leader);
            assert_eq!(request.tools.len(), 1);
            assert_eq!(request.tool_results.len(), 1);
            Ok(ProviderResponse {
                artifacts: vec![],
                tool_calls: vec![ToolCallRecord {
                    tool_call_id: ToolCallId::from("call-2"),
                    scope: request.scope.clone(),
                    task_id: request.task.task_id,
                    agent_id: AgentId::from("agent-root"),
                    tool_name: "exec_command".to_string(),
                    arguments_sha256: "hash".to_string(),
                    arguments_json: serde_json::json!({"cmd": "ls -la /app"}),
                    status: ToolCallStatus::Pending,
                    created_at_ms: 0,
                    resolved_at_ms: None,
                }],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone)]
    struct WorkerFindingToolContinuationProvider;

    #[async_trait::async_trait]
    impl ModelProvider for WorkerFindingToolContinuationProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            if request
                .task
                .objective
                .contains("continue root-visible tool execution")
            {
                assert!(
                    request
                        .input_text
                        .contains("Worker-agent findings available to the root leader")
                );
                assert!(request.input_text.contains("openssl x509"));
                return Ok(ProviderResponse {
                    artifacts: vec![],
                    tool_calls: vec![ToolCallRecord {
                        tool_call_id: ToolCallId::from("call-fixed"),
                        scope: request.scope.clone(),
                        task_id: request.task.task_id,
                        agent_id: AgentId::from("agent-root"),
                        tool_name: "exec_command".to_string(),
                        arguments_sha256: "hash".to_string(),
                        arguments_json: serde_json::json!({
                            "cmd": "openssl x509 -subject -noout -in /app/ssl/server.crt"
                        }),
                        status: ToolCallStatus::Pending,
                        created_at_ms: 0,
                        resolved_at_ms: None,
                    }],
                    usage: Default::default(),
                });
            }

            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                let mut children = vec![
                    model_child("fix-command", "replace opensslget with openssl x509"),
                    model_child("verify-command", "check corrected command arguments"),
                ];
                children.extend(model_children(
                    "command-repair-slice",
                    target.saturating_sub(children.len()),
                    "analyze command repair evidence",
                ));
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-worker-finding-tool"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason: "workers should identify the corrected command".to_string(),
                            children,
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 512,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }

            let text = match request.task.role {
                AgentRole::Leader => "root observed failed command".to_string(),
                AgentRole::Worker => {
                    "Use: openssl x509 -subject -noout -in /app/ssl/server.crt".to_string()
                }
                AgentRole::Synthesizer => "unexpected synthesizer".to_string(),
                AgentRole::ReasoningSummarizer | AgentRole::Verifier => String::new(),
            };
            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text,
                }],
                tool_calls: vec![],
                usage: Default::default(),
            })
        }
    }

    #[derive(Clone)]
    struct BudgetOverflowProvider;

    #[async_trait::async_trait]
    impl ModelProvider for BudgetOverflowProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from("oversized"),
                    scope: request.scope,
                    text: "oversized final answer".to_string(),
                }],
                tool_calls: vec![],
                usage: ProviderUsage {
                    input_tokens: 27_747,
                    output_tokens: 4_954,
                },
            })
        }
    }

    #[derive(Clone)]
    struct DenseStructuredProvider;

    #[async_trait::async_trait]
    impl ModelProvider for DenseStructuredProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-structured"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason: "format-sensitive synthesis needs one worker finding"
                                .to_string(),
                            children: model_children(
                                "structured-child",
                                target,
                                "produce the structured finding to preserve",
                            ),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 256,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }
            let text = match request.task.role {
                AgentRole::ReasoningSummarizer => "structured finding",
                AgentRole::Synthesizer => {
                    "<doc><section>###Heading<details><hr>FieldOne: alphaFieldTwo: beta</details></section></doc>"
                }
                AgentRole::Worker => "structured finding",
                AgentRole::Leader | AgentRole::Verifier => "root output",
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
                AgentRole::Leader if request.task.objective.contains("orchestration plan") => {
                    let target = target_child_agents(&request);
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-tool-visibility"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: request.task.task_id,
                                reason: "verify tool visibility".to_string(),
                                children: model_children(
                                    "tool-visibility-child",
                                    target,
                                    "child should not see client tools",
                                ),
                                expected_artifacts: vec![ArtifactKind::Text],
                                budget_request: BudgetRequest {
                                    max_tokens: 128,
                                    max_tool_calls: 0,
                                },
                            },
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Leader
                | AgentRole::Worker
                | AgentRole::ReasoningSummarizer
                | AgentRole::Synthesizer
                | AgentRole::Verifier => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: "done".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
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
                AgentRole::Worker
                | AgentRole::ReasoningSummarizer
                | AgentRole::Synthesizer
                | AgentRole::Verifier => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: request.task.objective,
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
            }
        }
    }

    #[derive(Clone)]
    struct ParallelProbeProvider {
        children: usize,
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ParallelProbeProvider {
        fn new(children: usize) -> Self {
            Self {
                children,
                in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                max_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for ParallelProbeProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-parallel-probe"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: request.task.task_id,
                            reason: "probe bounded backend fanout".to_string(),
                            children: (0..self.children)
                                .map(|index| SubtaskSpec {
                                    task_id: TaskId::from(format!("probe-child-{index}")),
                                    parent_task_id: Some(TaskId::from("root")),
                                    spawn_depth: 1,
                                    role: AgentRole::Worker,
                                    objective: format!("probe work {index}"),
                                    input_artifact_refs: vec![],
                                    expected_outputs: vec![ArtifactKind::Text],
                                    allowed_capabilities: CapabilitySet::from([Capability::Text]),
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
                }),
                AgentRole::Worker => {
                    let current = self
                        .in_flight
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        + 1;
                    self.max_in_flight
                        .fetch_max(current, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    self.in_flight
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
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
                AgentRole::ReasoningSummarizer | AgentRole::Synthesizer | AgentRole::Verifier => {
                    Ok(ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("parallel-probe-final"),
                            scope: request.scope,
                            text: "parallel probe final".to_string(),
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
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let target = target_child_agents(&request);
                return Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::SpawnPlan {
                        id: ArtifactId::from("spawn-plan-probe"),
                        scope: request.scope.clone(),
                        plan: SpawnPlan {
                            parent_task_id: TaskId::from("root"),
                            reason: "probe only needs to inspect policy instructions".to_string(),
                            children: model_children(
                                "probe-child",
                                target,
                                "observe policy instruction slice",
                            ),
                            expected_artifacts: vec![ArtifactKind::Text],
                            budget_request: BudgetRequest {
                                max_tokens: 0,
                                max_tool_calls: 0,
                            },
                        },
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                });
            }
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
                    let target = target_parallel_agents(&request.system_instructions);
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
                AgentRole::ReasoningSummarizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("coverage-summary"),
                        scope: request.scope,
                        text: request.input_text,
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

    #[derive(Clone)]
    struct UsageFanoutProvider;

    #[async_trait::async_trait]
    impl ModelProvider for UsageFanoutProvider {
        async fn invoke(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            match request.task.role {
                AgentRole::Leader => {
                    let target = target_child_agents(&request);
                    Ok(ProviderResponse {
                        artifacts: vec![
                            AgentArtifact::Text {
                                id: ArtifactId::from("input"),
                                scope: request.scope.clone(),
                                text: "seed".to_string(),
                            },
                            AgentArtifact::SpawnPlan {
                                id: ArtifactId::from("spawn-plan-usage"),
                                scope: request.scope.clone(),
                                plan: SpawnPlan {
                                    parent_task_id: request.task.task_id,
                                    reason: "measure child usage".to_string(),
                                    children: (0..target)
                                        .map(|index| {
                                            child(
                                                &format!("usage-child-{index:02}"),
                                                &TaskId::from("root"),
                                                1,
                                                &request.scope,
                                            )
                                        })
                                        .collect(),
                                    expected_artifacts: vec![ArtifactKind::Text],
                                    budget_request: BudgetRequest {
                                        max_tokens: 256,
                                        max_tool_calls: 0,
                                    },
                                },
                            },
                        ],
                        tool_calls: vec![],
                        usage: ProviderUsage {
                            input_tokens: 3,
                            output_tokens: 2,
                        },
                    })
                }
                AgentRole::Worker => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: request.task.objective,
                    }],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 5,
                        output_tokens: 7,
                    },
                }),
                AgentRole::ReasoningSummarizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("usage-summary"),
                        scope: request.scope,
                        text: request.input_text,
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => Ok(ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("text-synthesizer"),
                        scope: request.scope,
                        text: "usage final answer".to_string(),
                    }],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 11,
                        output_tokens: 13,
                    },
                }),
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

    fn target_child_agents(request: &ProviderRequest) -> usize {
        request
            .task
            .objective
            .split("exactly ")
            .nth(1)
            .and_then(|tail| tail.split_whitespace().next())
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| target_parallel_agents(&request.system_instructions))
            .max(1)
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

    fn model_child(id: &str, objective: &str) -> SubtaskSpec {
        SubtaskSpec {
            task_id: TaskId::from(id),
            parent_task_id: Some(TaskId::from("root")),
            spawn_depth: 1,
            role: AgentRole::Worker,
            objective: objective.to_string(),
            input_artifact_refs: vec![],
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([Capability::Text]),
            limits: AgentLimits::default(),
        }
    }

    fn model_children(prefix: &str, count: usize, objective_prefix: &str) -> Vec<SubtaskSpec> {
        (0..count)
            .map(|index| {
                model_child(
                    &format!("{prefix}-{index:02}"),
                    &format!("{objective_prefix} {index}"),
                )
            })
            .collect()
    }
}
