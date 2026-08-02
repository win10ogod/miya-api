use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

id_type!(RequestId);
id_type!(TenantId);
id_type!(ConversationFingerprint);
id_type!(TaskId);
id_type!(AgentId);
id_type!(ArtifactId);
id_type!(ToolCallId);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IsolationKey {
    pub tenant_id: TenantId,
    pub request_id: RequestId,
    pub conversation_fingerprint: ConversationFingerprint,
}

impl IsolationKey {
    pub fn new(tenant_id: &str, request_id: &str, conversation_fingerprint: &str) -> Self {
        Self {
            tenant_id: TenantId::from(tenant_id),
            request_id: RequestId::from(request_id),
            conversation_fingerprint: ConversationFingerprint::from(conversation_fingerprint),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLimits {
    pub max_spawn_depth: u8,
    pub max_agents_per_request: u16,
    pub max_parallel_agents: u16,
    pub max_tool_calls_per_agent: u16,
    pub max_total_tool_calls: u16,
    pub max_tokens_per_agent: u32,
    pub max_total_tokens: u32,
    pub request_timeout_ms: u64,
    pub agent_timeout_ms: u64,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_spawn_depth: 2,
            max_agents_per_request: 8,
            max_parallel_agents: 4,
            max_tool_calls_per_agent: 4,
            max_total_tool_calls: 16,
            max_tokens_per_agent: 2_048,
            max_total_tokens: 16_384,
            request_timeout_ms: 60_000,
            agent_timeout_ms: 20_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentLimits {
    pub max_tokens: u32,
    pub max_tool_calls: u16,
    pub timeout_ms: u64,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_tokens: 2_048,
            max_tool_calls: 4,
            timeout_ms: 20_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetRequest {
    pub max_tokens: u32,
    pub max_tool_calls: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetSummary {
    pub token_budget: u32,
    pub token_used: u32,
    pub tool_call_budget: u16,
    pub tool_calls_used: u16,
}

impl Default for BudgetSummary {
    fn default() -> Self {
        Self {
            token_budget: ExecutionLimits::default().max_total_tokens,
            token_used: 0,
            tool_call_budget: ExecutionLimits::default().max_total_tool_calls,
            tool_calls_used: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaArtifact {
    pub id: ArtifactId,
    pub scope: IsolationKey,
    pub media_type: String,
    pub source: MediaSource,
    pub sha256: String,
    pub byte_len: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    Base64 { data: String },
    DataUrl { data_url: String },
    RemoteUrl { url: String },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Text,
    Media,
    SpawnPlan,
    ToolCall,
    ToolResult,
    Verification,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub scope: IsolationKey,
    pub artifact_id: ArtifactId,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Leader,
    Worker,
    Verifier,
    ReasoningSummarizer,
    Synthesizer,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Text,
    Image,
    ToolCall,
    Spawn,
    Verify,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub capabilities: BTreeSet<Capability>,
}

impl<const N: usize> From<[Capability; N]> for CapabilitySet {
    fn from(value: [Capability; N]) -> Self {
        Self {
            capabilities: BTreeSet::from(value),
        }
    }
}

impl CapabilitySet {
    pub fn contains(&self, capability: &Capability) -> bool {
        self.capabilities.contains(capability)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtaskSpec {
    pub task_id: TaskId,
    pub parent_task_id: Option<TaskId>,
    pub spawn_depth: u8,
    pub role: AgentRole,
    pub objective: String,
    pub input_artifact_refs: Vec<ArtifactRef>,
    pub expected_outputs: Vec<ArtifactKind>,
    pub allowed_capabilities: CapabilitySet,
    pub limits: AgentLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskGraph {
    pub root_task_id: TaskId,
    pub tasks: BTreeMap<TaskId, SubtaskSpec>,
    pub dependencies: BTreeMap<TaskId, BTreeSet<TaskId>>,
}

impl TaskGraph {
    pub fn new(root_task_id: TaskId) -> Self {
        Self {
            root_task_id,
            tasks: BTreeMap::new(),
            dependencies: BTreeMap::new(),
        }
    }

    pub fn insert_task(&mut self, task: SubtaskSpec) {
        if let Some(parent) = task.parent_task_id.clone() {
            self.dependencies
                .entry(task.task_id.clone())
                .or_default()
                .insert(parent);
        }
        self.tasks.insert(task.task_id.clone(), task);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentInvocation {
    pub agent_id: AgentId,
    pub task_id: TaskId,
    pub provider_model: String,
    pub input_artifact_refs: Vec<ArtifactRef>,
    pub tool_policy: ToolPolicy,
    pub output_contract: OutputContract,
    pub limits: AgentLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPolicy {
    None,
    ClientSideAllowed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputContract {
    pub expected_outputs: Vec<ArtifactKind>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnPlan {
    pub parent_task_id: TaskId,
    pub reason: String,
    pub children: Vec<SubtaskSpec>,
    pub expected_artifacts: Vec<ArtifactKind>,
    pub budget_request: BudgetRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Named {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool_call_id: ToolCallId,
    pub scope: IsolationKey,
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub tool_name: String,
    pub arguments_json: serde_json::Value,
    pub arguments_sha256: String,
    pub status: ToolCallStatus,
    pub created_at_ms: u64,
    pub resolved_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    Resolved,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResultRecord {
    pub tool_call_id: ToolCallId,
    pub scope: IsolationKey,
    pub result_json: serde_json::Value,
    pub result_sha256: String,
    pub status: ToolResultStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    Accepted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub request_id: RequestId,
    pub passed: bool,
    pub issues: Vec<VerificationIssue>,
    pub artifact_coverage: Vec<ArtifactCoverage>,
    pub unresolved_tool_calls: Vec<ToolCallId>,
    pub budget_summary: BudgetSummary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptedAgentState {
    pub task_id: TaskId,
    pub algorithm: String,
    pub nonce: String,
    pub ciphertext: String,
    pub aad: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationIssue {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactCoverage {
    pub artifact_id: ArtifactId,
    pub kind: ArtifactKind,
    pub covered: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentArtifact {
    Text {
        id: ArtifactId,
        scope: IsolationKey,
        text: String,
    },
    Media(MediaArtifact),
    SpawnPlan {
        id: ArtifactId,
        scope: IsolationKey,
        plan: SpawnPlan,
    },
    ToolCall(ToolCallRecord),
    ToolResult(ToolResultRecord),
    Verification {
        id: ArtifactId,
        scope: IsolationKey,
        report: VerificationReport,
    },
}

impl AgentArtifact {
    pub fn id(&self) -> ArtifactId {
        match self {
            Self::Text { id, .. } => id.clone(),
            Self::Media(media) => media.id.clone(),
            Self::SpawnPlan { id, .. } => id.clone(),
            Self::ToolCall(record) => ArtifactId(record.tool_call_id.0.clone()),
            Self::ToolResult(record) => ArtifactId(record.tool_call_id.0.clone()),
            Self::Verification { id, .. } => id.clone(),
        }
    }

    pub fn scope(&self) -> &IsolationKey {
        match self {
            Self::Text { scope, .. } => scope,
            Self::Media(media) => &media.scope,
            Self::SpawnPlan { scope, .. } => scope,
            Self::ToolCall(record) => &record.scope,
            Self::ToolResult(record) => &record.scope,
            Self::Verification { scope, .. } => scope,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRequest {
    pub request_id: RequestId,
    pub tenant_id: TenantId,
    pub conversation_fingerprint: ConversationFingerprint,
    pub source_format: SourceFormat,
    pub model: String,
    pub messages: Vec<NormalizedMessage>,
    pub media_artifacts: Vec<MediaArtifact>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub tool_results: Vec<ToolResultRecord>,
    pub stream: bool,
    pub thinking_enabled: bool,
    pub thinking_format: ThinkingFormat,
    pub reasoning_effort: ReasoningEffort,
    #[serde(default)]
    pub public_reasoning_enabled: bool,
    #[serde(default)]
    pub provider_options: serde_json::Value,
    pub metadata: serde_json::Value,
}

impl NormalizedRequest {
    pub fn isolation_key(&self) -> IsolationKey {
        IsolationKey {
            tenant_id: self.tenant_id.clone(),
            request_id: self.request_id.clone(),
            conversation_fingerprint: self.conversation_fingerprint.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    OpenAIChat,
    AnthropicMessages,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingFormat {
    #[default]
    Auto,
    QwenChatTemplate,
    QwenDashScope,
    GemmaSystemToken,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    pub fn max_agents(&self) -> u16 {
        match self {
            Self::None => 0,
            Self::Low => 4,
            Self::Medium => 16,
            Self::High => 32,
            Self::XHigh => 64,
        }
    }

    pub fn is_direct(&self) -> bool {
        matches!(self, Self::None)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedMessage {
    pub role: MessageRole,
    pub content: Vec<NormalizedContentPart>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NormalizedContentPart {
    Text {
        text: String,
    },
    Image {
        artifact_ref: ArtifactRef,
    },
    ProviderContent {
        source_format: SourceFormat,
        value: serde_json::Value,
    },
    ToolCall {
        tool_call_id: ToolCallId,
        tool_name: String,
        arguments_json: serde_json::Value,
    },
    ToolResult {
        tool_call_id: ToolCallId,
    },
}

pub fn crate_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_serializes_core_records() {
        let scope = IsolationKey::new("tenant-a", "request-a", "conversation-a");
        let artifact_id = ArtifactId::from("image-1");
        let media = MediaArtifact {
            id: artifact_id.clone(),
            scope: scope.clone(),
            media_type: "image/png".to_string(),
            source: MediaSource::DataUrl {
                data_url: "data:image/png;base64,AAAA".to_string(),
            },
            sha256: "hash".to_string(),
            byte_len: Some(4),
        };

        let artifact_json = serde_json::to_string(&media).unwrap();
        let decoded_media: MediaArtifact = serde_json::from_str(&artifact_json).unwrap();
        assert_eq!(decoded_media, media);

        let root_task = TaskId::from("task-root");
        let child_task = SubtaskSpec {
            task_id: TaskId::from("task-child"),
            parent_task_id: Some(root_task.clone()),
            spawn_depth: 1,
            role: AgentRole::Worker,
            objective: "inspect image".to_string(),
            input_artifact_refs: vec![ArtifactRef {
                scope: scope.clone(),
                artifact_id,
            }],
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([Capability::Text, Capability::Image]),
            limits: AgentLimits::default(),
        };

        let spawn_plan = SpawnPlan {
            parent_task_id: root_task.clone(),
            reason: "Need visual inspection".to_string(),
            children: vec![child_task.clone()],
            expected_artifacts: vec![ArtifactKind::Text],
            budget_request: BudgetRequest {
                max_tokens: 512,
                max_tool_calls: 0,
            },
        };

        let spawn_json = serde_json::to_string(&spawn_plan).unwrap();
        let decoded_spawn: SpawnPlan = serde_json::from_str(&spawn_json).unwrap();
        assert_eq!(decoded_spawn, spawn_plan);

        let tool_call = ToolCallRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope,
            task_id: root_task,
            agent_id: AgentId::from("agent-root"),
            tool_name: "lookup".to_string(),
            arguments_json: serde_json::json!({"q": "rust"}),
            arguments_sha256: "args-hash".to_string(),
            status: ToolCallStatus::Pending,
            created_at_ms: 1,
            resolved_at_ms: None,
        };

        let tool_json = serde_json::to_string(&tool_call).unwrap();
        let decoded_tool: ToolCallRecord = serde_json::from_str(&tool_json).unwrap();
        assert_eq!(decoded_tool, tool_call);

        let mut graph = TaskGraph::new(root_task_id("task-root"));
        graph.insert_task(child_task);
        assert_eq!(graph.tasks.len(), 1);
    }

    #[test]
    fn reasoning_effort_maps_to_agent_limits() {
        assert_eq!(ReasoningEffort::None.max_agents(), 0);
        assert_eq!(ReasoningEffort::Low.max_agents(), 4);
        assert_eq!(ReasoningEffort::Medium.max_agents(), 16);
        assert_eq!(ReasoningEffort::High.max_agents(), 32);
        assert_eq!(ReasoningEffort::XHigh.max_agents(), 64);
    }

    fn root_task_id(value: &str) -> TaskId {
        TaskId::from(value)
    }
}
