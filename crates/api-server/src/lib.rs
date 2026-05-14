use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    convert::Infallible,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use agent_kernel::{KernelOutput, KernelPolicy, KernelRunner, KernelTraceEvent, MockProvider};
use agent_protocol::*;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use context_store::{
    ContextAppendRecord, ContextAssembly, ContextAssemblyOptions, DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_CONTEXT_BYTES, DEFAULT_RECENT_TAIL_CHUNKS, SurrealKvContextStore,
};
use futures::{Stream, StreamExt, future::join_all};
use provider_anthropic::AnthropicProvider;
use provider_core::{
    ModelProvider, ProviderFinishReason, ProviderStream, ProviderStreamEvent, ProviderUsage,
};
use provider_openai::OpenAiProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

const MAX_BATCH_REQUESTS: usize = 64;
const DEFAULT_TENANT_ID: &str = "default";
const DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS: usize = 16;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ApiError {
    #[error("streaming is not supported by the MVP multi-agent kernel")]
    StreamUnsupported,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub tools: Vec<OpenAiTool>,
    #[serde(default)]
    pub tool_choice: serde_json::Value,
    #[serde(default)]
    pub functions: Vec<OpenAiFunctionTool>,
    #[serde(default)]
    pub function_call: serde_json::Value,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub thinking: serde_json::Value,
    #[serde(default)]
    pub reasoning: serde_json::Value,
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Value,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiChatBatchRequest {
    pub requests: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiCompletionRequest {
    pub model: String,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub thinking: serde_json::Value,
    #[serde(default)]
    pub reasoning: serde_json::Value,
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Value,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<OpenAiContent>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAiMessageToolCall>,
    #[serde(default)]
    pub function_call: Option<OpenAiLegacyFunctionCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiMessageToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: String,
    pub function: OpenAiMessageToolCallFunction,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiMessageToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiLegacyFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiTool {
    #[serde(default)]
    pub r#type: String,
    pub function: OpenAiFunctionTool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiFunctionTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub tool_choice: serde_json::Value,
    #[serde(default)]
    pub thinking: serde_json::Value,
    #[serde(default)]
    pub reasoning: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessagesBatchRequest {
    pub requests: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompatibilityError {
    pub error: CompatibilityErrorBody,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompatibilityErrorBody {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

#[derive(Clone)]
pub struct AppState {
    kernel: Arc<KernelRunner<Arc<dyn ModelProvider>>>,
    context: ApiContextManager,
    direct: DirectBackend,
    tenant_limiter: TenantConcurrencyLimiter,
    training_trace: TrainingTraceRecorder,
    public_reasoning_mode: PublicReasoningMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicReasoningMode {
    Request,
    Always,
    Never,
}

impl PublicReasoningMode {
    fn from_env() -> Self {
        for name in [
            "MIYA_PUBLIC_REASONING",
            "MIYA_PUBLIC_REASONING_MODE",
            "MULTI_AGENT_PUBLIC_REASONING",
            "PUBLIC_REASONING_MODE",
        ] {
            if let Ok(value) = std::env::var(name) {
                return Self::from_env_value(Some(&value));
            }
        }
        Self::from_env_value(None)
    }

    fn from_env_value(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("always" | "on" | "true" | "1" | "yes" | "include" | "enabled") => Self::Always,
            Some("never" | "off" | "false" | "0" | "no" | "strip" | "none" | "disabled") => {
                Self::Never
            }
            Some("request" | "requested" | "client" | "client_requested") => Self::Request,
            Some("auto" | "default") | None => Self::Always,
            Some(_) => Self::Always,
        }
    }

    fn resolve(self, request_requested: bool) -> bool {
        match self {
            Self::Request => request_requested,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::with_provider_context_and_direct(
            Arc::new(MockProvider),
            ApiContextManager::disabled(),
            DirectBackend::Mock,
        )
    }
}

impl AppState {
    pub fn with_provider(provider: Arc<dyn ModelProvider>) -> Self {
        Self::with_provider_context_and_direct(
            provider,
            ApiContextManager::disabled(),
            DirectBackend::Mock,
        )
    }

    pub fn with_provider_and_context(
        provider: Arc<dyn ModelProvider>,
        context: ApiContextManager,
    ) -> Self {
        Self::with_provider_context_and_direct(provider, context, DirectBackend::Mock)
    }

    pub fn with_provider_context_and_direct(
        provider: Arc<dyn ModelProvider>,
        context: ApiContextManager,
        direct: DirectBackend,
    ) -> Self {
        Self::with_provider_context_direct_and_tenant_limiter(
            provider,
            context,
            direct,
            TenantConcurrencyLimiter::disabled(),
        )
    }

    fn with_provider_context_direct_and_tenant_limiter(
        provider: Arc<dyn ModelProvider>,
        context: ApiContextManager,
        direct: DirectBackend,
        tenant_limiter: TenantConcurrencyLimiter,
    ) -> Self {
        Self::with_provider_context_direct_tenant_limiter_and_training_trace(
            provider,
            context,
            direct,
            tenant_limiter,
            TrainingTraceRecorder::disabled(),
        )
    }

    fn with_provider_context_direct_tenant_limiter_and_training_trace(
        provider: Arc<dyn ModelProvider>,
        context: ApiContextManager,
        direct: DirectBackend,
        tenant_limiter: TenantConcurrencyLimiter,
        training_trace: TrainingTraceRecorder,
    ) -> Self {
        Self::with_provider_context_direct_tenant_limiter_training_trace_and_policy(
            provider,
            context,
            direct,
            tenant_limiter,
            training_trace,
            KernelPolicy::default(),
        )
    }

    fn with_provider_context_direct_tenant_limiter_training_trace_and_policy(
        provider: Arc<dyn ModelProvider>,
        context: ApiContextManager,
        direct: DirectBackend,
        tenant_limiter: TenantConcurrencyLimiter,
        training_trace: TrainingTraceRecorder,
        policy: KernelPolicy,
    ) -> Self {
        Self {
            kernel: Arc::new(KernelRunner::new(provider, policy)),
            context,
            direct,
            tenant_limiter,
            training_trace,
            public_reasoning_mode: PublicReasoningMode::Request,
        }
    }
}

#[derive(Clone)]
struct TrainingTraceRecorder {
    path: Option<PathBuf>,
    lock: Arc<Mutex<()>>,
}

impl TrainingTraceRecorder {
    fn disabled() -> Self {
        Self {
            path: None,
            lock: Arc::new(Mutex::new(())),
        }
    }

    fn enabled_at(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            lock: Arc::new(Mutex::new(())),
        }
    }

    fn from_env() -> Self {
        let enabled = std::env::var("TRAINING_TRACE")
            .or_else(|_| std::env::var("MIYA_TRAINING_TRACE"))
            .map(|value| env_flag_enabled(&value))
            .unwrap_or(false);
        if !enabled {
            return Self::disabled();
        }

        let path = std::env::var("TRAINING_TRACE_PATH")
            .or_else(|_| std::env::var("MIYA_TRAINING_TRACE_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("logs").join("training-traces.jsonl"));
        Self::enabled_at(path)
    }

    fn capture_request(&self, request: &NormalizedRequest) -> Option<NormalizedRequest> {
        self.path.as_ref().map(|_| request.clone())
    }

    fn capture_openai_request(
        &self,
        request: &OpenAiChatRequest,
        request_context: &RequestContext,
    ) -> Option<NormalizedRequest> {
        self.path.as_ref()?;
        normalize_openai_chat_with_context(request.clone(), request_context).ok()
    }

    fn capture_anthropic_request(
        &self,
        request: &AnthropicMessagesRequest,
        request_context: &RequestContext,
    ) -> Option<NormalizedRequest> {
        self.path.as_ref()?;
        normalize_anthropic_messages_with_context(request.clone(), request_context).ok()
    }

    fn record_kernel(
        &self,
        request: Option<&NormalizedRequest>,
        output: &KernelOutput,
    ) -> Result<(), String> {
        let Some(request) = request else {
            return Ok(());
        };
        self.write(training_example_from_kernel_output(request, output))
    }

    fn record_direct(
        &self,
        request: Option<&NormalizedRequest>,
        assistant_text: Option<String>,
    ) -> Result<(), String> {
        let Some(request) = request else {
            return Ok(());
        };
        self.write(training_example_from_direct_output(request, assistant_text))
    }

    fn write(&self, value: serde_json::Value) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let _guard = self
            .lock
            .lock()
            .map_err(|_| "training trace lock poisoned".to_string())?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        writeln!(file, "{}", compact_json(value)).map_err(|error| error.to_string())
    }
}

#[derive(Clone)]
pub enum DirectBackend {
    Mock,
    OpenAi {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
    },
    Anthropic {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        api_version: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenAiToolResponseFormat {
    Tools,
    LegacyFunctions,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RequestContext {
    tenant_id: Option<String>,
    request_id: Option<String>,
    conversation_id: Option<String>,
}

impl RequestContext {
    fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            tenant_id: header_identity(
                headers,
                &[
                    "x-tenant-id",
                    "x-organization-id",
                    "x-project-id",
                    "x-user-id",
                ],
            ),
            request_id: header_identity(headers, &["x-request-id", "x-correlation-id"]),
            conversation_id: header_identity(headers, &["x-conversation-id", "x-thread-id"]),
        }
    }

    fn with_metadata_overrides(&self, metadata: &serde_json::Value) -> Self {
        Self {
            tenant_id: metadata_identity(
                metadata,
                &[
                    "tenant_id",
                    "tenant",
                    "organization_id",
                    "project_id",
                    "user_id",
                ],
            )
            .or_else(|| self.tenant_id.clone()),
            request_id: metadata_identity(metadata, &["request_id", "correlation_id"])
                .or_else(|| self.request_id.clone()),
            conversation_id: metadata_identity(
                metadata,
                &["conversation_id", "thread_id", "session_id"],
            )
            .or_else(|| self.conversation_id.clone()),
        }
    }

    fn tenant_id(&self) -> TenantId {
        TenantId::from(
            self.tenant_id
                .clone()
                .unwrap_or_else(|| DEFAULT_TENANT_ID.to_string()),
        )
    }

    fn request_id(&self) -> RequestId {
        RequestId::from(
            self.request_id
                .clone()
                .unwrap_or_else(|| Uuid::new_v4().to_string()),
        )
    }

    fn conversation_fingerprint(
        &self,
        source_format: &str,
        tenant_id: &TenantId,
        messages_debug: &str,
    ) -> ConversationFingerprint {
        let fingerprint_input = match &self.conversation_id {
            Some(conversation_id) => {
                format!(
                    "{source_format}:tenant={}:conversation={conversation_id}",
                    tenant_id.as_ref()
                )
            }
            None => format!(
                "{source_format}:tenant={}:messages={messages_debug}",
                tenant_id.as_ref()
            ),
        };
        ConversationFingerprint::from(sha256_hex(&fingerprint_input))
    }
}

#[derive(Clone, Debug)]
struct TelemetryContext {
    route: &'static str,
    model: String,
    source_format: &'static str,
    tenant_id: String,
    request_id: String,
    conversation_fingerprint: String,
    reasoning_effort: &'static str,
    stream: bool,
    batch_index: Option<usize>,
}

fn telemetry_context_from_normalized(
    route: &'static str,
    normalized: &NormalizedRequest,
    stream: bool,
    batch_index: Option<usize>,
) -> TelemetryContext {
    TelemetryContext {
        route,
        model: normalized.model.clone(),
        source_format: source_format_label(&normalized.source_format),
        tenant_id: normalized.tenant_id.as_ref().to_string(),
        request_id: normalized.request_id.as_ref().to_string(),
        conversation_fingerprint: normalized.conversation_fingerprint.as_ref().to_string(),
        reasoning_effort: reasoning_effort_label(&normalized.reasoning_effort),
        stream,
        batch_index,
    }
}

fn direct_telemetry_context(
    route: &'static str,
    model: String,
    source_format: &'static str,
    request_context: &RequestContext,
    stream: bool,
    batch_index: Option<usize>,
) -> TelemetryContext {
    let tenant_id = request_context.tenant_id();
    let request_id = request_context.request_id();
    let conversation_fingerprint =
        request_context.conversation_fingerprint(source_format, &tenant_id, "direct");
    TelemetryContext {
        route,
        model,
        source_format,
        tenant_id: tenant_id.as_ref().to_string(),
        request_id: request_id.as_ref().to_string(),
        conversation_fingerprint: conversation_fingerprint.as_ref().to_string(),
        reasoning_effort: "none",
        stream,
        batch_index,
    }
}

fn source_format_label(source_format: &SourceFormat) -> &'static str {
    match source_format {
        SourceFormat::OpenAIChat => "openai_chat",
        SourceFormat::AnthropicMessages => "anthropic_messages",
    }
}

fn reasoning_effort_label(reasoning_effort: &ReasoningEffort) -> &'static str {
    match reasoning_effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    }
}

fn header_identity(headers: &HeaderMap, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .and_then(identity_component)
    })
}

fn metadata_identity(metadata: &serde_json::Value, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        metadata
            .get(*name)
            .and_then(|value| match value {
                serde_json::Value::String(text) => Some(text.as_str()),
                _ => None,
            })
            .and_then(identity_component)
    })
}

fn identity_component(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let is_safe = trimmed.len() <= 128
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':' | '@'));
    if is_safe {
        Some(trimmed.to_string())
    } else {
        Some(format!("sha256-{}", &sha256_hex(trimmed)[..32]))
    }
}

fn env_flag_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

#[derive(Clone)]
struct TenantConcurrencyLimiter {
    max_per_tenant: Option<usize>,
    semaphores: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
}

impl TenantConcurrencyLimiter {
    fn disabled() -> Self {
        Self {
            max_per_tenant: None,
            semaphores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn with_max_per_tenant(max_per_tenant: usize) -> Self {
        if max_per_tenant == 0 {
            return Self::disabled();
        }
        Self {
            max_per_tenant: Some(max_per_tenant),
            semaphores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn from_env() -> Self {
        let env_value = std::env::var("TENANT_MAX_CONCURRENT_REQUESTS").ok();
        Self::from_env_value(env_value.as_deref())
    }

    fn from_env_value(value: Option<&str>) -> Self {
        let max = value
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS);
        Self::with_max_per_tenant(max)
    }

    async fn acquire(&self, tenant_id: &TenantId) -> Result<TenantConcurrencyPermit, String> {
        let Some(max_per_tenant) = self.max_per_tenant else {
            return Ok(TenantConcurrencyPermit { _permit: None });
        };
        let semaphore = {
            let mut semaphores = self
                .semaphores
                .lock()
                .map_err(|_| "tenant concurrency limiter lock poisoned".to_string())?;
            semaphores
                .entry(tenant_id.as_ref().to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(max_per_tenant)))
                .clone()
        };
        let permit = semaphore
            .acquire_owned()
            .await
            .map_err(|_| "tenant concurrency limiter was closed".to_string())?;
        Ok(TenantConcurrencyPermit {
            _permit: Some(permit),
        })
    }
}

struct TenantConcurrencyPermit {
    _permit: Option<OwnedSemaphorePermit>,
}

impl DirectBackend {
    fn from_env(provider_kind: &str) -> Result<Self, String> {
        match provider_kind {
            "mock" => Ok(Self::Mock),
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY is required".to_string())?;
                let base_url = std::env::var("OPENAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
                Ok(Self::OpenAi {
                    client: reqwest::Client::new(),
                    base_url: base_url.trim_end_matches('/').to_string(),
                    api_key,
                })
            }
            "anthropic" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| "ANTHROPIC_API_KEY is required".to_string())?;
                let base_url = std::env::var("ANTHROPIC_BASE_URL")
                    .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
                let api_version =
                    std::env::var("ANTHROPIC_VERSION").unwrap_or_else(|_| "2023-06-01".to_string());
                Ok(Self::Anthropic {
                    client: reqwest::Client::new(),
                    base_url: base_url.trim_end_matches('/').to_string(),
                    api_key,
                    api_version,
                })
            }
            other => Err(format!("unsupported direct provider={other}")),
        }
    }

    async fn openai_chat(&self, request: serde_json::Value) -> Result<serde_json::Value, String> {
        match self {
            Self::Mock => Ok(mock_direct_openai_response(&request)),
            Self::OpenAi {
                client,
                base_url,
                api_key,
            } => post_json(
                client
                    .post(format!("{base_url}/chat/completions"))
                    .bearer_auth(api_key),
                sanitize_direct_openai_request(request),
            )
            .await
            .map(strip_direct_openai_response),
            Self::Anthropic { .. } => Err(
                "reasoning.effort=none on /v1/chat/completions requires MULTI_AGENT_PROVIDER=openai"
                    .to_string(),
            ),
        }
    }

    async fn openai_chat_stream(&self, request: serde_json::Value) -> Result<Response, String> {
        match self {
            Self::Mock => Ok(openai_stream_response_from_completion(
                mock_direct_openai_response(&request),
            )),
            Self::OpenAi {
                client,
                base_url,
                api_key,
            } => {
                let response = client
                    .post(format!("{base_url}/chat/completions"))
                    .bearer_auth(api_key)
                    .json(&sanitize_direct_openai_request(request))
                    .send()
                    .await
                    .map_err(|error| error.to_string())?;
                let response = direct_response_or_error(response).await?;
                Ok(upstream_sse_response(response))
            }
            Self::Anthropic { .. } => Err(
                "reasoning.effort=none on /v1/chat/completions requires MULTI_AGENT_PROVIDER=openai"
                    .to_string(),
            ),
        }
    }

    async fn anthropic_messages(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        match self {
            Self::Mock => Ok(mock_direct_anthropic_response(&request)),
            Self::Anthropic {
                client,
                base_url,
                api_key,
                api_version,
            } => post_json(
                client
                    .post(format!("{base_url}/v1/messages"))
                    .header("x-api-key", api_key)
                    .header("anthropic-version", api_version),
                sanitize_direct_anthropic_request(request),
            )
            .await
            .map(strip_direct_anthropic_response),
            Self::OpenAi { .. } => Err(
                "reasoning.effort=none on /v1/messages requires MULTI_AGENT_PROVIDER=anthropic"
                    .to_string(),
            ),
        }
    }

    async fn anthropic_messages_stream(
        &self,
        request: serde_json::Value,
    ) -> Result<Response, String> {
        match self {
            Self::Mock => Ok(anthropic_stream_response_from_message(
                mock_direct_anthropic_response(&request),
            )),
            Self::Anthropic {
                client,
                base_url,
                api_key,
                api_version,
            } => {
                let response = client
                    .post(format!("{base_url}/v1/messages"))
                    .header("x-api-key", api_key)
                    .header("anthropic-version", api_version)
                    .json(&sanitize_direct_anthropic_request(request))
                    .send()
                    .await
                    .map_err(|error| error.to_string())?;
                let response = direct_response_or_error(response).await?;
                Ok(upstream_sse_response(response))
            }
            Self::OpenAi { .. } => Err(
                "reasoning.effort=none on /v1/messages requires MULTI_AGENT_PROVIDER=anthropic"
                    .to_string(),
            ),
        }
    }
}

fn upstream_sse_response(response: reqwest::Response) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(response.bytes_stream()))
        .unwrap_or_else(|error| internal_error_response(error.to_string()))
}

async fn post_json(
    builder: reqwest::RequestBuilder,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let response = builder
        .json(&body)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    direct_response_or_error(response)
        .await?
        .json::<serde_json::Value>()
        .await
        .map_err(|error| error.to_string())
}

async fn direct_response_or_error(
    response: reqwest::Response,
) -> Result<reqwest::Response, String> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let url = response.url().to_string();
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
    Err(format!(
        "upstream rejected request: HTTP {status} for {url}; body: {body}"
    ))
}

fn sanitize_direct_openai_request(mut request: serde_json::Value) -> serde_json::Value {
    strip_gateway_metadata(&mut request);
    strip_gateway_reasoning_effort(&mut request);
    strip_gateway_public_reasoning_options(&mut request);
    disable_qwen_thinking_by_default(&mut request);
    normalize_direct_openai_tools(&mut request);
    adapt_gemma_direct_named_tool_choice(&mut request);
    enable_gemma_thinking_prompt_by_default(&mut request);
    request
}

fn sanitize_direct_anthropic_request(mut request: serde_json::Value) -> serde_json::Value {
    strip_gateway_metadata(&mut request);
    strip_gateway_reasoning_effort(&mut request);
    strip_gateway_public_reasoning_options(&mut request);
    request
}

fn strip_gateway_metadata(request: &mut serde_json::Value) {
    let Some(object) = request.as_object_mut() else {
        return;
    };
    let Some(metadata) = object.get("metadata").cloned() else {
        return;
    };
    if let Some(metadata) = sanitized_provider_metadata(&metadata) {
        object.insert("metadata".to_string(), metadata);
    } else {
        object.remove("metadata");
    }
}

fn strip_gateway_reasoning_effort(request: &mut serde_json::Value) {
    let Some(reasoning) = request
        .get_mut("reasoning")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };
    reasoning.remove("effort");
    if reasoning.is_empty()
        && let Some(object) = request.as_object_mut()
    {
        object.remove("reasoning");
    }
}

fn strip_gateway_public_reasoning_options(request: &mut serde_json::Value) {
    let Some(object) = request.as_object_mut() else {
        return;
    };
    for key in [
        "include_reasoning",
        "include_thinking",
        "show_reasoning",
        "return_reasoning",
        "reasoning_content",
    ] {
        object.remove(key);
    }
}

fn disable_qwen_thinking_by_default(request: &mut serde_json::Value) {
    let model = request
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !model.contains("qwen") {
        return;
    }
    let Some(object) = request.as_object_mut() else {
        return;
    };
    if object.contains_key("chat_template_kwargs")
        || object.contains_key("enable_thinking")
        || object.contains_key("thinking")
    {
        return;
    }
    object.insert(
        "chat_template_kwargs".to_string(),
        serde_json::json!({
            "enable_thinking": false,
            "preserve_thinking": false
        }),
    );
}

fn normalize_direct_openai_tools(request: &mut serde_json::Value) {
    let Some(tools) = request
        .get_mut("tools")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for tool in tools {
        let Some(function) = tool
            .get_mut("function")
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };

        let parameters = function
            .entry("parameters".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if !parameters.is_object() {
            *parameters = serde_json::json!({});
        }

        let object = parameters.as_object_mut().expect("parameters object");
        object
            .entry("type".to_string())
            .or_insert_with(|| serde_json::Value::String("object".to_string()));
        object
            .entry("properties".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if !object
            .get("properties")
            .is_some_and(serde_json::Value::is_object)
        {
            object.insert("properties".to_string(), serde_json::json!({}));
        }
    }
}

fn adapt_gemma_direct_named_tool_choice(request: &mut serde_json::Value) {
    let model = request
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    if !is_gemma_system_token_model(model) {
        return;
    }

    let Some(name) = direct_openai_named_tool_choice_name(request).map(str::to_string) else {
        return;
    };

    request["tool_choice"] = serde_json::Value::String("required".to_string());
    if let Some(tools) = request
        .get_mut("tools")
        .and_then(serde_json::Value::as_array_mut)
    {
        tools.retain(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(serde_json::Value::as_str)
                == Some(name.as_str())
        });
    }
}

fn direct_openai_named_tool_choice_name(request: &serde_json::Value) -> Option<&str> {
    let tool_choice = request.get("tool_choice")?;
    if tool_choice.get("type").and_then(serde_json::Value::as_str) != Some("function") {
        return None;
    }
    tool_choice
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.trim().is_empty())
}

fn enable_gemma_thinking_prompt_by_default(request: &mut serde_json::Value) {
    let model = request
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    if !is_gemma_system_token_model(model) || openai_request_has_think_token(request) {
        return;
    }

    let Some(messages) = request
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return;
    };
    let think_prompt = "<|think|>\n";
    if let Some(first) = messages.first_mut()
        && first.get("role").and_then(|value| value.as_str()) == Some("system")
    {
        match first.get_mut("content") {
            Some(serde_json::Value::String(content)) => {
                content.insert_str(0, think_prompt);
                return;
            }
            Some(serde_json::Value::Array(parts)) => {
                parts.insert(
                    0,
                    serde_json::json!({
                        "type": "text",
                        "text": think_prompt
                    }),
                );
                return;
            }
            _ => {}
        }
    }

    messages.insert(
        0,
        serde_json::json!({
            "role": "system",
            "content": think_prompt
        }),
    );
}

fn openai_request_has_think_token(request: &serde_json::Value) -> bool {
    request
        .get("messages")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content"))
        .any(content_has_think_token)
}

fn content_has_think_token(content: &serde_json::Value) -> bool {
    match content {
        serde_json::Value::String(text) => text.contains("<|think|>"),
        serde_json::Value::Array(parts) => parts.iter().any(|part| {
            part.get("text")
                .and_then(|value| value.as_str())
                .is_some_and(|text| text.contains("<|think|>"))
        }),
        _ => false,
    }
}

fn mock_direct_openai_response(request: &serde_json::Value) -> serde_json::Value {
    let model = request
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("mock");
    let text = direct_openai_user_text(request);
    serde_json::json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": format!("direct backend response: {text}")
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })
}

fn mock_direct_anthropic_response(request: &serde_json::Value) -> serde_json::Value {
    let model = request
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("mock");
    let text = direct_anthropic_user_text(request);
    serde_json::json!({
        "id": format!("msg_{}", Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{
            "type": "text",
            "text": format!("direct backend response: {text}")
        }],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0
        }
    })
}

fn strip_direct_openai_response(mut response: serde_json::Value) -> serde_json::Value {
    if let Some(choices) = response
        .get_mut("choices")
        .and_then(|value| value.as_array_mut())
    {
        for choice in choices {
            if let Some(message) = choice
                .get_mut("message")
                .and_then(|value| value.as_object_mut())
            {
                message.remove("reasoning");
                message.remove("reasoning_content");
                if let Some(content) = message.get_mut("content")
                    && let Some(text) = content.as_str()
                {
                    *content = serde_json::Value::String(strip_direct_thinking_markup(text));
                }
            }
        }
    }
    response
}

fn strip_direct_anthropic_response(mut response: serde_json::Value) -> serde_json::Value {
    if let Some(content) = response
        .get_mut("content")
        .and_then(|value| value.as_array_mut())
    {
        content.retain(|block| {
            !matches!(
                block.get("type").and_then(|value| value.as_str()),
                Some("thinking" | "redacted_thinking")
            )
        });
    }
    response
}

fn strip_direct_thinking_markup(content: &str) -> String {
    let text = if let Some((_, after)) = content.rsplit_once("<channel|>") {
        after
    } else if let Some((_, after)) = content.rsplit_once("</think>") {
        after
    } else {
        content
    };
    strip_direct_generation_wrappers(text)
}

fn strip_direct_generation_wrappers(content: &str) -> String {
    let mut text = content.trim().to_string();

    for marker in ["<start_of_turn>model", "<start_of_turn>assistant"] {
        if let Some((_, after)) = text.rsplit_once(marker) {
            text = after.to_string();
            break;
        }
    }

    if let Some((before, _)) = text.split_once("<end_of_turn>") {
        text = before.to_string();
    }

    for token in [
        "<bos>",
        "<eos>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|start_of_turn|>",
        "<|end_of_turn|>",
    ] {
        text = text.replace(token, "");
    }

    text.trim().to_string()
}

fn direct_openai_user_text(request: &serde_json::Value) -> String {
    request
        .get("messages")
        .and_then(|value| value.as_array())
        .and_then(|messages| {
            messages
                .iter()
                .rev()
                .find(|message| message.get("role").and_then(|role| role.as_str()) == Some("user"))
        })
        .map(|message| direct_content_text(message.get("content")))
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "ok".to_string())
}

fn direct_anthropic_user_text(request: &serde_json::Value) -> String {
    direct_openai_user_text(request)
}

fn direct_content_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

#[derive(Clone)]
pub struct ApiContextManager {
    store: Option<Arc<SurrealKvContextStore>>,
}

impl ApiContextManager {
    pub fn disabled() -> Self {
        Self { store: None }
    }

    pub fn surreal_kv(store: SurrealKvContextStore) -> Self {
        Self {
            store: Some(Arc::new(store)),
        }
    }

    fn from_env() -> Result<Self, String> {
        if std::env::var("CONTEXT_STORE")
            .map(|value| value == "disabled")
            .unwrap_or(false)
        {
            return Ok(Self::disabled());
        }
        let path = std::env::var("CONTEXT_STORE_PATH")
            .unwrap_or_else(|_| ".multi-agent-context/surrealkv".to_string());
        SurrealKvContextStore::open(path)
            .map(Self::surreal_kv)
            .map_err(|error| error.to_string())
    }

    async fn prepare(
        &self,
        normalized: &mut NormalizedRequest,
    ) -> Result<Option<PreparedContext>, String> {
        let Some(config) = ContextRuntimeConfig::from_metadata(&normalized.metadata) else {
            return Ok(None);
        };
        let tenant_id = normalized.tenant_id.as_ref().to_string();
        let inbound_records = context_records_from_request(normalized);
        let Some(store) = &self.store else {
            return Ok(Some(PreparedContext {
                config,
                tenant_id,
                inbound_records,
                report: ContextUsageReport::disabled(),
            }));
        };

        let query = config
            .query
            .clone()
            .or_else(|| last_user_text(normalized))
            .filter(|query| !query.trim().is_empty());
        let cache_namespace =
            context_cache_namespace(normalized, config.cache_namespace.as_deref());
        let assembly = store
            .assemble(
                normalized.tenant_id.as_ref(),
                &config.context_id,
                ContextAssemblyOptions {
                    query,
                    cache_namespace,
                    rewind_revision: config.rewind_revision,
                    max_context_bytes: config.max_context_bytes,
                    max_chunks: config.max_chunks,
                    recent_tail_chunks: config.recent_tail_chunks,
                    cache_enabled: config.cache_enabled,
                },
            )
            .await
            .map_err(|error| error.to_string())?;

        if !assembly.text.is_empty() {
            normalized.messages.insert(
                0,
                NormalizedMessage {
                    role: MessageRole::System,
                    content: vec![NormalizedContentPart::Text {
                        text: format!(
                            "Persistent context retrieved from SurrealKV context_id={} revision={}:\n{}",
                            config.context_id, assembly.revision, assembly.text
                        ),
                    }],
                },
            );
        }

        Ok(Some(PreparedContext {
            tenant_id,
            inbound_records,
            report: ContextUsageReport::from_assembly(&assembly),
            config,
        }))
    }

    async fn record_response(
        &self,
        prepared: Option<&PreparedContext>,
        output: &KernelOutput,
    ) -> Result<Option<u64>, String> {
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        if !prepared.config.append_current {
            return Ok(None);
        }
        let Some(store) = &self.store else {
            return Ok(None);
        };

        let mut records = prepared.inbound_records.clone();
        if !output.final_text.trim().is_empty() {
            records.push(ContextAppendRecord {
                role: "assistant".to_string(),
                text: output.final_text.clone(),
            });
        }
        if records.is_empty() {
            return Ok(None);
        }

        let head = store
            .append(&prepared.tenant_id, &prepared.config.context_id, records)
            .await
            .map_err(|error| error.to_string())?;
        Ok(Some(head.latest_revision))
    }
}

#[derive(Clone, Debug)]
struct PreparedContext {
    config: ContextRuntimeConfig,
    tenant_id: String,
    inbound_records: Vec<ContextAppendRecord>,
    report: ContextUsageReport,
}

#[derive(Clone, Debug)]
struct ContextRuntimeConfig {
    context_id: String,
    query: Option<String>,
    cache_namespace: Option<String>,
    rewind_revision: Option<u64>,
    max_context_bytes: usize,
    max_chunks: usize,
    recent_tail_chunks: usize,
    cache_enabled: bool,
    append_current: bool,
    include_report: bool,
}

impl ContextRuntimeConfig {
    fn from_metadata(metadata: &serde_json::Value) -> Option<Self> {
        let context = metadata.get("context")?;
        if context
            .get("enabled")
            .and_then(|value| value.as_bool())
            .is_some_and(|enabled| !enabled)
        {
            return None;
        }
        let context_id = context
            .get("id")
            .or_else(|| context.get("context_id"))
            .and_then(|value| value.as_str())?
            .trim()
            .to_string();
        if context_id.is_empty() {
            return None;
        }

        Some(Self {
            context_id,
            query: context
                .get("query")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            cache_namespace: context
                .get("cache_namespace")
                .or_else(|| context.get("namespace"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
            rewind_revision: context
                .get("rewind_revision")
                .or_else(|| context.get("revision"))
                .and_then(|value| value.as_u64()),
            max_context_bytes: context
                .get("max_context_bytes")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_MAX_CONTEXT_BYTES)
                .min(DEFAULT_MAX_CONTEXT_BYTES),
            max_chunks: context
                .get("max_chunks")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_MAX_CHUNKS)
                .min(DEFAULT_MAX_CHUNKS),
            recent_tail_chunks: context
                .get("recent_tail_chunks")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_RECENT_TAIL_CHUNKS)
                .min(DEFAULT_MAX_CHUNKS),
            cache_enabled: context
                .get("cache")
                .and_then(|value| value.as_bool())
                .unwrap_or(true),
            append_current: context
                .get("append")
                .or_else(|| context.get("append_current"))
                .and_then(|value| value.as_bool())
                .unwrap_or(true),
            include_report: context
                .get("include_report")
                .or_else(|| metadata.get("include_context_report"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        })
    }
}

fn context_cache_namespace(request: &NormalizedRequest, explicit: Option<&str>) -> String {
    if let Some(namespace) = explicit.and_then(normalize_context_cache_namespace) {
        return namespace;
    }
    generated_context_cache_namespace(request)
}

fn generated_context_cache_namespace(request: &NormalizedRequest) -> String {
    let system_hash = short_hash(&system_text_from_request(request));
    let tools_hash = short_hash(
        &serde_json::json!({
            "tools": &request.tools,
            "tool_choice": &request.tool_choice,
            "parallel_tool_calls": request.parallel_tool_calls,
        })
        .to_string(),
    );
    let provider_options_hash = short_hash(&request.provider_options.to_string());
    let thinking_enabled = if request.thinking_enabled {
        "on"
    } else {
        "off"
    };
    normalize_context_cache_namespace(&format!(
        "{}|model:{}|thinking:{}:{}|tools:{}|provider:{}|system:{}",
        source_format_label(&request.source_format),
        request.model,
        thinking_enabled,
        thinking_format_label(&request.thinking_format),
        tools_hash,
        provider_options_hash,
        system_hash
    ))
    .unwrap_or_else(|| "default".to_string())
}

fn normalize_context_cache_namespace(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.chars().take(256).collect())
    }
}

fn short_hash(value: &str) -> String {
    sha256_hex(value).chars().take(16).collect()
}

fn thinking_format_label(format: &ThinkingFormat) -> &'static str {
    match format {
        ThinkingFormat::Auto => "auto",
        ThinkingFormat::QwenChatTemplate => "qwen_chat_template",
        ThinkingFormat::QwenDashScope => "qwen_dashscope",
        ThinkingFormat::GemmaSystemToken => "gemma_system_token",
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct ContextUsageReport {
    enabled: bool,
    context_id: Option<String>,
    cache_namespace: Option<String>,
    revision: u64,
    included_chunks: usize,
    included_bytes: usize,
    cache_hit: bool,
    base_cache_revision: Option<u64>,
    tail_chunks: usize,
    stored_revision: Option<u64>,
}

impl ContextUsageReport {
    fn disabled() -> Self {
        Self::default()
    }

    fn from_assembly(assembly: &ContextAssembly) -> Self {
        Self {
            enabled: true,
            context_id: Some(assembly.context_id.clone()),
            cache_namespace: Some(assembly.cache_namespace.clone()),
            revision: assembly.revision,
            included_chunks: assembly.included_chunks,
            included_bytes: assembly.included_bytes,
            cache_hit: assembly.cache_hit,
            base_cache_revision: assembly.base_cache_revision,
            tail_chunks: assembly.tail_chunks,
            stored_revision: None,
        }
    }
}

pub fn build_router() -> Router {
    build_router_with_state(AppState::default())
}

pub fn build_router_from_env() -> Result<Router, String> {
    let provider_kind =
        std::env::var("MULTI_AGENT_PROVIDER").unwrap_or_else(|_| "mock".to_string());
    let provider = provider_from_kind(&provider_kind)?;
    let context = ApiContextManager::from_env()?;
    let direct = DirectBackend::from_env(&provider_kind)?;
    let tenant_limiter = TenantConcurrencyLimiter::from_env();
    let training_trace = TrainingTraceRecorder::from_env();
    let kernel_policy = kernel_policy_from_env()?;
    let mut state = AppState::with_provider_context_direct_tenant_limiter_training_trace_and_policy(
        provider,
        context,
        direct,
        tenant_limiter,
        training_trace,
        kernel_policy,
    );
    state.public_reasoning_mode = PublicReasoningMode::from_env();
    Ok(build_router_with_state(state))
}

fn kernel_policy_from_env() -> Result<KernelPolicy, String> {
    let mut policy = KernelPolicy::default();
    if let Some(max_parallel_agents) = env_u16(&[
        "MULTI_AGENT_MAX_PARALLEL_AGENTS",
        "MIYA_MAX_PARALLEL_AGENTS",
        "MAX_PARALLEL_AGENTS",
    ])? {
        if max_parallel_agents == 0 {
            return Err(
                "MAX_PARALLEL_AGENTS must be at least 1; use reasoning.effort=none for direct mode"
                    .to_string(),
            );
        }
        policy.limits.max_parallel_agents = max_parallel_agents.min(64);
    }
    Ok(policy)
}

fn env_u16(names: &[&str]) -> Result<Option<u16>, String> {
    for name in names {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        let parsed = value
            .trim()
            .parse::<u16>()
            .map_err(|_| format!("{name} must be an integer between 1 and 65535"))?;
        return Ok(Some(parsed));
    }
    Ok(None)
}

pub fn build_router_with_provider(provider: Arc<dyn ModelProvider>) -> Router {
    build_router_with_state(AppState::with_provider(provider))
}

pub fn build_router_with_provider_and_context(
    provider: Arc<dyn ModelProvider>,
    context: ApiContextManager,
) -> Router {
    build_router_with_state(AppState::with_provider_and_context(provider, context))
}

pub fn build_router_with_provider_and_tenant_limit(
    provider: Arc<dyn ModelProvider>,
    max_per_tenant: usize,
) -> Router {
    build_router_with_state(AppState::with_provider_context_direct_and_tenant_limiter(
        provider,
        ApiContextManager::disabled(),
        DirectBackend::Mock,
        TenantConcurrencyLimiter::with_max_per_tenant(max_per_tenant),
    ))
}

pub fn build_router_with_provider_context_and_direct(
    provider: Arc<dyn ModelProvider>,
    context: ApiContextManager,
    direct: DirectBackend,
) -> Router {
    build_router_with_state(AppState::with_provider_context_and_direct(
        provider, context, direct,
    ))
}

fn build_router_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/v1/health", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/v1/v1/models", get(models))
        .route("/completions", post(completions))
        .route("/v1/completions", post(completions))
        .route("/v1/v1/completions", post(completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/v1/chat/completions", post(chat_completions))
        .route("/chat/completions/batch", post(chat_completions_batch))
        .route("/v1/chat/completions/batch", post(chat_completions_batch))
        .route(
            "/v1/v1/chat/completions/batch",
            post(chat_completions_batch),
        )
        .route("/messages", post(messages))
        .route("/v1/messages", post(messages))
        .route("/v1/v1/messages", post(messages))
        .route("/messages/batch", post(messages_batch))
        .route("/v1/messages/batch", post(messages_batch))
        .route("/v1/v1/messages/batch", post(messages_batch))
        .with_state(state)
}

fn provider_from_kind(provider_kind: &str) -> Result<Arc<dyn ModelProvider>, String> {
    match provider_kind {
        "mock" => Ok(Arc::new(MockProvider)),
        "openai" => OpenAiProvider::from_env()
            .map(|provider| Arc::new(provider) as Arc<dyn ModelProvider>)
            .map_err(|error| error.to_string()),
        "anthropic" => AnthropicProvider::from_env()
            .map(|provider| Arc::new(provider) as Arc<dyn ModelProvider>)
            .map_err(|error| error.to_string()),
        other => Err(format!(
            "unsupported MULTI_AGENT_PROVIDER={other}; expected mock, openai, or anthropic"
        )),
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn models() -> Json<serde_json::Value> {
    let data = configured_model_ids()
        .into_iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "owned_by": "multi-agent-api"
            })
        })
        .collect::<Vec<_>>();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

fn configured_model_ids() -> Vec<String> {
    let raw = ["MULTI_AGENT_MODELS", "MIYA_MODELS", "OPENAI_MODELS"]
        .iter()
        .find_map(|name| std::env::var(name).ok());
    configured_model_ids_from(raw.as_deref())
}

fn configured_model_ids_from(raw: Option<&str>) -> Vec<String> {
    let mut base_ids = Vec::new();

    if let Some(raw) = raw {
        if raw.trim_start().starts_with('[') {
            if let Ok(values) = serde_json::from_str::<Vec<String>>(raw) {
                for value in values {
                    push_model_id(&mut base_ids, value);
                }
            }
        } else {
            for value in raw.split([',', ';', '\n']) {
                push_model_id(&mut base_ids, value);
            }
        }
    }

    push_model_id(&mut base_ids, "mock");

    let mut ids = Vec::new();
    for base_id in base_ids {
        push_model_id(&mut ids, &base_id);
        push_effort_model_aliases(&mut ids, &base_id);
    }
    ids
}

fn push_model_id(ids: &mut Vec<String>, value: impl AsRef<str>) {
    let value = value.as_ref().trim();
    if value.is_empty() || ids.iter().any(|existing| existing == value) {
        return;
    }
    ids.push(value.to_string());
}

fn model_list_from_env(names: &[&str]) -> Vec<String> {
    let raw = names.iter().find_map(|name| std::env::var(name).ok());
    parse_model_list(raw.as_deref())
}

fn parse_model_list(raw: Option<&str>) -> Vec<String> {
    let mut ids = Vec::new();
    let Some(raw) = raw else {
        return ids;
    };

    if raw.trim_start().starts_with('[') {
        if let Ok(values) = serde_json::from_str::<Vec<String>>(raw) {
            for value in values {
                push_model_id(&mut ids, value);
            }
        }
    } else {
        for value in raw.split([',', ';', '\n']) {
            push_model_id(&mut ids, value);
        }
    }
    ids
}

fn push_effort_model_aliases(ids: &mut Vec<String>, base_model: &str) {
    if !model_supports_effort_aliases(base_model) || split_effort_model_alias(base_model).is_some()
    {
        return;
    }

    for (suffix, _) in effort_model_suffixes() {
        push_model_id(ids, format!("{base_model}{suffix}"));
    }
}

fn model_supports_effort_aliases(model: &str) -> bool {
    !model.trim().eq_ignore_ascii_case("mock")
}

fn raw_request_with_provider_model(
    mut raw_request: serde_json::Value,
    request_model: &str,
) -> serde_json::Value {
    let provider_model = provider_model_for_request_model(request_model);
    if provider_model != request_model {
        raw_request["model"] = serde_json::Value::String(provider_model);
    }
    raw_request
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let completion_request = match parse_openai_completion_request(raw_request) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = completion_request.stream;
    let mut request = match completion_request.into_chat_request() {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    request.stream = false;
    let include_public_reasoning = state
        .public_reasoning_mode
        .resolve(openai_public_reasoning_requested(&request));

    let request_context =
        RequestContext::from_headers(&headers).with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let tenant_permit = match state.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return internal_error_response(error),
    };
    let model = request.model.clone();
    let mut normalized = match normalize_openai_chat_with_context(request, &request_context) {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let thinking_format = normalized.thinking_format.clone();
    let telemetry_context =
        telemetry_context_from_normalized("openai.completions", &normalized, stream, None);
    let training_request = state.training_trace.capture_request(&normalized);

    if stream && !requires_full_orchestration_before_stream(&normalized, include_public_reasoning) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => response_with_tenant_permit(
                format_openai_completion_provider_stream_response(
                    model,
                    provider_stream,
                    telemetry_context,
                ),
                tenant_permit,
            ),
            Err(error) => internal_error_response(error.to_string()),
        };
    }

    match state.kernel.run(normalized).await {
        Ok(output) => {
            let context_report =
                match finalize_context_report(&state.context, &prepared_context, &output).await {
                    Ok(report) => report,
                    Err(error) => return internal_error_response(error),
                };
            emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
            if let Err(error) = state
                .training_trace
                .record_kernel(training_request.as_ref(), &output)
            {
                log_training_trace_error(error);
            }
            if stream {
                return format_openai_completion_stream_response(
                    model,
                    output,
                    include_public_reasoning,
                    thinking_format,
                );
            }
            let mut response = format_openai_completion_response(
                model,
                output,
                include_public_reasoning,
                thinking_format,
            );
            attach_context_report(&mut response, prepared_context.as_ref(), context_report);
            Json(response).into_response()
        }
        Err(error) => internal_error_response(error.to_string()),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let mut request = match parse_openai_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = request.stream;
    request.stream = false;
    let include_public_reasoning = state
        .public_reasoning_mode
        .resolve(openai_public_reasoning_requested(&request));
    let request_context =
        RequestContext::from_headers(&headers).with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let tenant_permit = match state.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return internal_error_response(error),
    };
    let direct_training_request = state
        .training_trace
        .capture_openai_request(&request, &request_context);
    if matches!(openai_reasoning_effort(&request), Ok(ReasoningEffort::None)) {
        let provider_raw_request = raw_request_with_provider_model(raw_request, &request.model);
        if stream {
            return match state.direct.openai_chat_stream(provider_raw_request).await {
                Ok(response) => response_with_tenant_permit(response, tenant_permit),
                Err(error) => internal_error_response(error),
            };
        }
        return match state.direct.openai_chat(provider_raw_request).await {
            Ok(value) => {
                let telemetry_context = direct_telemetry_context(
                    "openai.chat_completions",
                    request.model.clone(),
                    "openai_chat",
                    &request_context,
                    false,
                    None,
                );
                emit_direct_telemetry(&telemetry_context, response_usage(&value));
                if let Err(error) = state.training_trace.record_direct(
                    direct_training_request.as_ref(),
                    openai_response_text(&value),
                ) {
                    log_training_trace_error(error);
                }
                Json(value).into_response()
            }
            Err(error) => internal_error_response(error),
        };
    }

    let model = request.model.clone();
    let tool_response_format = openai_tool_response_format(&request);
    let mut normalized = match normalize_openai_chat_with_context(request, &request_context) {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);
    let telemetry_context =
        telemetry_context_from_normalized("openai.chat_completions", &normalized, stream, None);
    let training_request = state.training_trace.capture_request(&normalized);

    if stream && !requires_full_orchestration_before_stream(&normalized, include_public_reasoning) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => response_with_tenant_permit(
                format_openai_provider_stream_response(
                    model,
                    provider_stream,
                    tool_response_format,
                    telemetry_context,
                ),
                tenant_permit,
            ),
            Err(error) => internal_error_response(error.to_string()),
        };
    }

    match state.kernel.run(normalized).await {
        Ok(output) => {
            let context_report =
                match finalize_context_report(&state.context, &prepared_context, &output).await {
                    Ok(report) => report,
                    Err(error) => return internal_error_response(error),
                };
            emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
            if let Err(error) = state
                .training_trace
                .record_kernel(training_request.as_ref(), &output)
            {
                log_training_trace_error(error);
            }
            if stream {
                return format_openai_stream_response(
                    model,
                    output,
                    tool_response_format,
                    include_public_reasoning,
                );
            }
            let mut response = format_openai_response(
                model,
                output,
                include_encrypted_state,
                tool_response_format,
                include_public_reasoning,
            );
            attach_context_report(&mut response, prepared_context.as_ref(), context_report);
            Json(response).into_response()
        }
        Err(error) => internal_error_response(error.to_string()),
    }
}

async fn chat_completions_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(batch): Json<OpenAiChatBatchRequest>,
) -> Response {
    if let Err(error) = validate_batch_len(batch.requests.len()) {
        return api_error_response(error);
    }

    let deps = BatchRequestDeps {
        kernel: state.kernel.clone(),
        context: state.context.clone(),
        direct: state.direct.clone(),
        tenant_limiter: state.tenant_limiter.clone(),
        training_trace: state.training_trace.clone(),
        public_reasoning_mode: state.public_reasoning_mode,
    };
    let request_context = RequestContext::from_headers(&headers);
    let data = join_all(
        batch
            .requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let deps = deps.clone();
                let request_context = request_context.clone();
                async move { run_openai_batch_item(deps, request_context, index, request).await }
            }),
    )
    .await;

    Json(serde_json::json!({
        "object": "chat.completion.batch",
        "data": data
    }))
    .into_response()
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let mut request = match parse_anthropic_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = request.stream;
    request.stream = false;
    let include_public_reasoning = state
        .public_reasoning_mode
        .resolve(anthropic_public_reasoning_requested(&request));
    let request_context =
        RequestContext::from_headers(&headers).with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let tenant_permit = match state.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return internal_error_response(error),
    };
    let direct_training_request = state
        .training_trace
        .capture_anthropic_request(&request, &request_context);
    if matches!(
        anthropic_reasoning_effort(&request),
        Ok(ReasoningEffort::None)
    ) {
        let provider_raw_request = raw_request_with_provider_model(raw_request, &request.model);
        if stream {
            return match state
                .direct
                .anthropic_messages_stream(provider_raw_request)
                .await
            {
                Ok(response) => response_with_tenant_permit(response, tenant_permit),
                Err(error) => internal_error_response(error),
            };
        }
        return match state.direct.anthropic_messages(provider_raw_request).await {
            Ok(value) => {
                let telemetry_context = direct_telemetry_context(
                    "anthropic.messages",
                    request.model.clone(),
                    "anthropic_messages",
                    &request_context,
                    false,
                    None,
                );
                emit_direct_telemetry(&telemetry_context, response_usage(&value));
                if let Err(error) = state.training_trace.record_direct(
                    direct_training_request.as_ref(),
                    anthropic_response_text(&value),
                ) {
                    log_training_trace_error(error);
                }
                Json(value).into_response()
            }
            Err(error) => internal_error_response(error),
        };
    }

    let model = request.model.clone();
    let mut normalized = match normalize_anthropic_messages_with_context(request, &request_context)
    {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);
    let telemetry_context =
        telemetry_context_from_normalized("anthropic.messages", &normalized, stream, None);
    let training_request = state.training_trace.capture_request(&normalized);

    if stream && !requires_full_orchestration_before_stream(&normalized, include_public_reasoning) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => response_with_tenant_permit(
                format_anthropic_provider_stream_response(
                    model,
                    provider_stream,
                    telemetry_context,
                ),
                tenant_permit,
            ),
            Err(error) => internal_error_response(error.to_string()),
        };
    }

    match state.kernel.run(normalized).await {
        Ok(output) => {
            let context_report =
                match finalize_context_report(&state.context, &prepared_context, &output).await {
                    Ok(report) => report,
                    Err(error) => return internal_error_response(error),
                };
            emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
            if let Err(error) = state
                .training_trace
                .record_kernel(training_request.as_ref(), &output)
            {
                log_training_trace_error(error);
            }
            if stream {
                return format_anthropic_stream_response(model, output, include_public_reasoning);
            }
            let mut response = format_anthropic_response(
                model,
                output,
                include_encrypted_state,
                include_public_reasoning,
            );
            attach_context_report(&mut response, prepared_context.as_ref(), context_report);
            Json(response).into_response()
        }
        Err(error) => internal_error_response(error.to_string()),
    }
}

async fn messages_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(batch): Json<AnthropicMessagesBatchRequest>,
) -> Response {
    if let Err(error) = validate_batch_len(batch.requests.len()) {
        return api_error_response(error);
    }

    let deps = BatchRequestDeps {
        kernel: state.kernel.clone(),
        context: state.context.clone(),
        direct: state.direct.clone(),
        tenant_limiter: state.tenant_limiter.clone(),
        training_trace: state.training_trace.clone(),
        public_reasoning_mode: state.public_reasoning_mode,
    };
    let request_context = RequestContext::from_headers(&headers);
    let data = join_all(
        batch
            .requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let deps = deps.clone();
                let request_context = request_context.clone();
                async move { run_anthropic_batch_item(deps, request_context, index, request).await }
            }),
    )
    .await;

    Json(serde_json::json!({
        "type": "message_batch",
        "data": data
    }))
    .into_response()
}

#[derive(Clone)]
struct BatchRequestDeps {
    kernel: Arc<KernelRunner<Arc<dyn ModelProvider>>>,
    context: ApiContextManager,
    direct: DirectBackend,
    tenant_limiter: TenantConcurrencyLimiter,
    training_trace: TrainingTraceRecorder,
    public_reasoning_mode: PublicReasoningMode,
}

async fn run_openai_batch_item(
    deps: BatchRequestDeps,
    request_context: RequestContext,
    index: usize,
    raw_request: serde_json::Value,
) -> serde_json::Value {
    let request = match parse_openai_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return batch_api_error(index, error),
    };
    if request.stream {
        return batch_api_error(index, ApiError::StreamUnsupported);
    }
    let include_public_reasoning = deps
        .public_reasoning_mode
        .resolve(openai_public_reasoning_requested(&request));
    let request_context = request_context.with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let _tenant_permit = match deps.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return batch_kernel_error(index, error),
    };
    let direct_training_request = deps
        .training_trace
        .capture_openai_request(&request, &request_context);
    if matches!(openai_reasoning_effort(&request), Ok(ReasoningEffort::None)) {
        let provider_raw_request = raw_request_with_provider_model(raw_request, &request.model);
        return match deps.direct.openai_chat(provider_raw_request).await {
            Ok(response) => {
                let telemetry_context = direct_telemetry_context(
                    "openai.chat_completions.batch_item",
                    request.model.clone(),
                    "openai_chat",
                    &request_context,
                    false,
                    Some(index),
                );
                emit_direct_telemetry(&telemetry_context, response_usage(&response));
                if let Err(error) = deps.training_trace.record_direct(
                    direct_training_request.as_ref(),
                    openai_response_text(&response),
                ) {
                    log_training_trace_error(error);
                }
                batch_success(index, response)
            }
            Err(error) => batch_kernel_error(index, error),
        };
    }

    let model = request.model.clone();
    let tool_response_format = openai_tool_response_format(&request);
    let mut normalized = match normalize_openai_chat_with_context(request, &request_context) {
        Ok(normalized) => normalized,
        Err(error) => return batch_api_error(index, error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match deps.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return batch_kernel_error(index, error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);
    let telemetry_context = telemetry_context_from_normalized(
        "openai.chat_completions.batch_item",
        &normalized,
        false,
        Some(index),
    );
    let training_request = deps.training_trace.capture_request(&normalized);

    match deps.kernel.run(normalized).await {
        Ok(output) => {
            match finalize_context_report(&deps.context, &prepared_context, &output).await {
                Ok(context_report) => {
                    emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
                    if let Err(error) = deps
                        .training_trace
                        .record_kernel(training_request.as_ref(), &output)
                    {
                        log_training_trace_error(error);
                    }
                    let mut response = format_openai_response(
                        model,
                        output,
                        include_encrypted_state,
                        tool_response_format,
                        include_public_reasoning,
                    );
                    attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                    batch_success(index, response)
                }
                Err(error) => batch_kernel_error(index, error),
            }
        }
        Err(error) => batch_kernel_error(index, error.to_string()),
    }
}

async fn run_anthropic_batch_item(
    deps: BatchRequestDeps,
    request_context: RequestContext,
    index: usize,
    raw_request: serde_json::Value,
) -> serde_json::Value {
    let request = match parse_anthropic_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return batch_api_error(index, error),
    };
    if request.stream {
        return batch_api_error(index, ApiError::StreamUnsupported);
    }
    let include_public_reasoning = deps
        .public_reasoning_mode
        .resolve(anthropic_public_reasoning_requested(&request));
    let request_context = request_context.with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let _tenant_permit = match deps.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return batch_kernel_error(index, error),
    };
    let direct_training_request = deps
        .training_trace
        .capture_anthropic_request(&request, &request_context);
    if matches!(
        anthropic_reasoning_effort(&request),
        Ok(ReasoningEffort::None)
    ) {
        let provider_raw_request = raw_request_with_provider_model(raw_request, &request.model);
        return match deps.direct.anthropic_messages(provider_raw_request).await {
            Ok(response) => {
                let telemetry_context = direct_telemetry_context(
                    "anthropic.messages.batch_item",
                    request.model.clone(),
                    "anthropic_messages",
                    &request_context,
                    false,
                    Some(index),
                );
                emit_direct_telemetry(&telemetry_context, response_usage(&response));
                if let Err(error) = deps.training_trace.record_direct(
                    direct_training_request.as_ref(),
                    anthropic_response_text(&response),
                ) {
                    log_training_trace_error(error);
                }
                batch_success(index, response)
            }
            Err(error) => batch_kernel_error(index, error),
        };
    }

    let model = request.model.clone();
    let mut normalized = match normalize_anthropic_messages_with_context(request, &request_context)
    {
        Ok(normalized) => normalized,
        Err(error) => return batch_api_error(index, error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match deps.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return batch_kernel_error(index, error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);
    let telemetry_context = telemetry_context_from_normalized(
        "anthropic.messages.batch_item",
        &normalized,
        false,
        Some(index),
    );
    let training_request = deps.training_trace.capture_request(&normalized);

    match deps.kernel.run(normalized).await {
        Ok(output) => {
            match finalize_context_report(&deps.context, &prepared_context, &output).await {
                Ok(context_report) => {
                    emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
                    if let Err(error) = deps
                        .training_trace
                        .record_kernel(training_request.as_ref(), &output)
                    {
                        log_training_trace_error(error);
                    }
                    let mut response = format_anthropic_response(
                        model,
                        output,
                        include_encrypted_state,
                        include_public_reasoning,
                    );
                    attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                    batch_success(index, response)
                }
                Err(error) => batch_kernel_error(index, error),
            }
        }
        Err(error) => batch_kernel_error(index, error.to_string()),
    }
}

fn validate_batch_len(len: usize) -> Result<(), ApiError> {
    if len == 0 {
        return Err(ApiError::InvalidRequest(
            "batch requests must contain at least one request".to_string(),
        ));
    }
    if len > MAX_BATCH_REQUESTS {
        return Err(ApiError::InvalidRequest(format!(
            "batch requests must not exceed {MAX_BATCH_REQUESTS} items"
        )));
    }
    Ok(())
}

fn batch_success(index: usize, response: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "response": response,
        "error": null
    })
}

fn batch_api_error(index: usize, error: ApiError) -> serde_json::Value {
    let (_, code, message) = api_error_parts(error);
    batch_error(index, code, "invalid_request_error", message)
}

fn batch_kernel_error(index: usize, message: String) -> serde_json::Value {
    batch_error(index, "kernel_error", "server_error", message)
}

fn batch_error(
    index: usize,
    code: &'static str,
    error_type: &'static str,
    message: String,
) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "response": null,
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    })
}

async fn finalize_context_report(
    context: &ApiContextManager,
    prepared_context: &Option<PreparedContext>,
    output: &KernelOutput,
) -> Result<Option<ContextUsageReport>, String> {
    let Some(prepared) = prepared_context else {
        return Ok(None);
    };
    let mut report = prepared.report.clone();
    report.stored_revision = context.record_response(Some(prepared), output).await?;
    Ok(Some(report))
}

fn attach_context_report(
    response: &mut serde_json::Value,
    prepared_context: Option<&PreparedContext>,
    report: Option<ContextUsageReport>,
) {
    let Some(prepared) = prepared_context else {
        return;
    };
    if prepared.config.include_report {
        response["context_cache"] = serde_json::json!(report.unwrap_or_default());
    }
}

fn context_records_from_request(request: &NormalizedRequest) -> Vec<ContextAppendRecord> {
    request
        .messages
        .iter()
        .filter_map(|message| {
            let text = message_text(message);
            if text.trim().is_empty() {
                return None;
            }
            Some(ContextAppendRecord {
                role: match &message.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                }
                .to_string(),
                text,
            })
        })
        .collect()
}

fn last_user_text(request: &NormalizedRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::User)
        .map(message_text)
        .filter(|text| !text.trim().is_empty())
}

fn message_text(message: &NormalizedMessage) -> String {
    message
        .content
        .iter()
        .map(|part| match part {
            NormalizedContentPart::Text { text } => text.as_str(),
            NormalizedContentPart::Image { .. } => "[image]",
            NormalizedContentPart::ToolCall { .. } => "[tool_call]",
            NormalizedContentPart::ToolResult { .. } => "[tool_result]",
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn training_example_from_kernel_output(
    request: &NormalizedRequest,
    output: &KernelOutput,
) -> serde_json::Value {
    let mut conversations = training_conversations_from_request(request);
    let mut seen_tool_call_ids = BTreeSet::new();

    for event in &output.trace_events {
        match event {
            KernelTraceEvent::SpawnPlan {
                reason, children, ..
            } => {
                conversations.push(training_turn(
                    "function_call",
                    compact_json(serde_json::json!({
                        "name": "spawn_agent",
                        "arguments": {
                            "reason": reason,
                            "children": children.iter().map(|child| serde_json::json!({
                                "task_id": child.task_id.as_ref(),
                                "role": format!("{:?}", child.role),
                                "objective": child.objective,
                            })).collect::<Vec<_>>()
                        }
                    })),
                ));
            }
            KernelTraceEvent::AgentOutput {
                task_id,
                role,
                text_outputs,
                tool_calls,
                ..
            } => {
                for call in tool_calls {
                    seen_tool_call_ids.insert(call.tool_call_id.as_ref().to_string());
                    conversations.push(training_tool_call_turn(call));
                }
                if matches!(role, AgentRole::Worker) && !text_outputs.is_empty() {
                    conversations.push(training_turn(
                        "observation",
                        compact_json(serde_json::json!({
                            "task_id": task_id.as_ref(),
                            "role": format!("{:?}", role),
                            "outputs": text_outputs
                        })),
                    ));
                }
            }
            KernelTraceEvent::AgentInput { .. } => {}
        }
    }

    for call in &output.tool_calls {
        if seen_tool_call_ids.insert(call.tool_call_id.as_ref().to_string()) {
            conversations.push(training_tool_call_turn(call));
        }
    }

    for result in &request.tool_results {
        conversations.push(training_tool_result_turn(result));
    }

    if !output.final_text.trim().is_empty() {
        conversations.push(training_turn("gpt", output.final_text.clone()));
    }

    training_example_json(
        request,
        conversations,
        output
            .trace_events
            .iter()
            .any(|event| matches!(event, KernelTraceEvent::SpawnPlan { .. })),
    )
}

fn training_example_from_direct_output(
    request: &NormalizedRequest,
    assistant_text: Option<String>,
) -> serde_json::Value {
    let mut conversations = training_conversations_from_request(request);
    for result in &request.tool_results {
        conversations.push(training_tool_result_turn(result));
    }
    if let Some(text) = assistant_text.filter(|text| !text.trim().is_empty()) {
        conversations.push(training_turn("gpt", text));
    }
    training_example_json(request, conversations, false)
}

fn training_example_json(
    request: &NormalizedRequest,
    conversations: Vec<serde_json::Value>,
    include_spawn_tool: bool,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "conversations": conversations
    });

    let system = system_text_from_request(request);
    if !system.trim().is_empty() {
        value["system"] = serde_json::Value::String(system);
    }

    let tools = training_tools_json(request, include_spawn_tool);
    if let Some(tools) = tools {
        value["tools"] = serde_json::Value::String(tools);
    }

    value
}

fn training_conversations_from_request(request: &NormalizedRequest) -> Vec<serde_json::Value> {
    let mut turns = Vec::new();
    for message in &request.messages {
        match message.role {
            MessageRole::System => {}
            MessageRole::User => {
                let text = message_text(message);
                if !text.trim().is_empty() {
                    turns.push(training_turn("human", text));
                }
            }
            MessageRole::Assistant => {
                for part in &message.content {
                    match part {
                        NormalizedContentPart::Text { text } if !text.trim().is_empty() => {
                            turns.push(training_turn("gpt", text.clone()));
                        }
                        NormalizedContentPart::ToolCall {
                            tool_call_id,
                            tool_name,
                            arguments_json,
                        } => turns.push(training_turn(
                            "function_call",
                            compact_json(serde_json::json!({
                                "tool_call_id": tool_call_id.as_ref(),
                                "name": tool_name,
                                "arguments": arguments_json
                            })),
                        )),
                        _ => {}
                    }
                }
            }
            MessageRole::Tool => {
                for part in &message.content {
                    if let NormalizedContentPart::ToolResult { tool_call_id } = part
                        && let Some(result) = request
                            .tool_results
                            .iter()
                            .find(|result| result.tool_call_id == *tool_call_id)
                    {
                        turns.push(training_tool_result_turn(result));
                    }
                }
            }
        }
    }
    turns
}

fn training_turn(from: &str, value: String) -> serde_json::Value {
    serde_json::json!({
        "from": from,
        "value": value
    })
}

fn training_tool_call_turn(call: &ToolCallRecord) -> serde_json::Value {
    training_turn(
        "function_call",
        compact_json(serde_json::json!({
            "tool_call_id": call.tool_call_id.as_ref(),
            "name": call.tool_name,
            "arguments": call.arguments_json
        })),
    )
}

fn training_tool_result_turn(result: &ToolResultRecord) -> serde_json::Value {
    training_turn(
        "observation",
        compact_json(serde_json::json!({
            "tool_call_id": result.tool_call_id.as_ref(),
            "result": result.result_json,
            "status": result.status
        })),
    )
}

fn system_text_from_request(request: &NormalizedRequest) -> String {
    request
        .messages
        .iter()
        .filter(|message| message.role == MessageRole::System)
        .map(message_text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn training_tools_json(request: &NormalizedRequest, include_spawn_tool: bool) -> Option<String> {
    let mut tools = request
        .tools
        .iter()
        .map(|tool| serde_json::json!(tool))
        .collect::<Vec<_>>();

    if include_spawn_tool {
        tools.push(serde_json::json!({
            "name": "spawn_agent",
            "description": "Dispatch bounded sub-agent tasks during deterministic orchestration.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "role": {"type": "string"},
                                "objective": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }));
    }

    if tools.is_empty() {
        None
    } else {
        Some(compact_json(serde_json::Value::Array(tools)))
    }
}

fn openai_response_text(value: &serde_json::Value) -> Option<String> {
    value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .map(str::to_string)
}

fn anthropic_response_text(value: &serde_json::Value) -> Option<String> {
    value
        .get("content")
        .and_then(|content| content.as_array())
        .map(|content| {
            content
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|value| value.as_str()) == Some("text") {
                        block.get("text").and_then(|text| text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
}

fn api_error_response(error: ApiError) -> Response {
    let (status, code, message) = api_error_parts(error);

    (
        status,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message,
                r#type: "invalid_request_error".to_string(),
                code: code.to_string(),
            },
        }),
    )
        .into_response()
}

fn api_error_parts(error: ApiError) -> (StatusCode, &'static str, String) {
    let (status, code, message) = match error {
        ApiError::StreamUnsupported => (
            StatusCode::BAD_REQUEST,
            "stream_unsupported",
            "streaming is not supported by the MVP multi-agent kernel".to_string(),
        ),
        ApiError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, "invalid_request", message),
    };
    (status, code, message)
}

fn internal_error_response(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message,
                r#type: "server_error".to_string(),
                code: "kernel_error".to_string(),
            },
        }),
    )
        .into_response()
}

fn log_training_trace_error(error: String) {
    eprintln!("training trace error: {error}");
}

fn response_with_tenant_permit(
    mut response: Response,
    permit: TenantConcurrencyPermit,
) -> Response {
    response.extensions_mut().insert(Arc::new(permit));
    response
}

fn sse_response(body: String) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(body))
        .unwrap_or_else(|error| internal_error_response(error.to_string()))
}

fn sse_stream_response<S>(stream: S) -> Response
where
    S: Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|error| internal_error_response(error.to_string()))
}

fn sse_data(value: serde_json::Value) -> String {
    format!("data: {}\n\n", compact_json(value))
}

fn sse_event(event: &str, value: serde_json::Value) -> String {
    format!("event: {event}\ndata: {}\n\n", compact_json(value))
}

fn compact_json(value: serde_json::Value) -> String {
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
}

fn emit_telemetry(record: serde_json::Value) {
    println!("{}", compact_json(record));
}

fn emit_kernel_telemetry(
    context: &TelemetryContext,
    output: &KernelOutput,
    context_report: Option<&ContextUsageReport>,
) {
    emit_telemetry(kernel_telemetry_record(context, output, context_report));
}

fn emit_direct_telemetry(context: &TelemetryContext, usage: ProviderUsage) {
    emit_telemetry(direct_telemetry_record(context, &usage));
}

fn kernel_telemetry_record(
    context: &TelemetryContext,
    output: &KernelOutput,
    context_report: Option<&ContextUsageReport>,
) -> serde_json::Value {
    let usage = usage_telemetry_json(&output.usage);
    serde_json::json!({
        "event": "api_usage",
        "event_version": 1,
        "timestamp_ms": telemetry_timestamp_ms(),
        "route": context.route,
        "model": context.model,
        "source_format": context.source_format,
        "tenant_id": context.tenant_id,
        "request_id": context.request_id,
        "conversation_fingerprint": context.conversation_fingerprint,
        "reasoning_effort": context.reasoning_effort,
        "stream": context.stream,
        "batch_index": context.batch_index,
        "direct_passthrough": false,
        "usage": usage,
        "input_tokens": output.usage.input_tokens,
        "output_tokens": output.usage.output_tokens,
        "total_tokens": output.usage.input_tokens.saturating_add(output.usage.output_tokens),
        "provider_call_count": output.provider_call_count,
        "task_count": output.task_graph.tasks.len(),
        "child_agent_count": output.encrypted_subagent_state.len(),
        "tool_call_count": output.tool_calls.len(),
        "final_text_bytes": output.final_text.len(),
        "verification": {
            "passed": output.verification.passed,
            "issue_count": output.verification.issues.len(),
            "unresolved_tool_call_count": output.verification.unresolved_tool_calls.len(),
            "budget_summary": &output.verification.budget_summary
        },
        "context": context_report
            .map(|report| serde_json::json!(report))
            .unwrap_or(serde_json::Value::Null),
    })
}

fn direct_telemetry_record(context: &TelemetryContext, usage: &ProviderUsage) -> serde_json::Value {
    serde_json::json!({
        "event": "api_usage",
        "event_version": 1,
        "timestamp_ms": telemetry_timestamp_ms(),
        "route": context.route,
        "model": context.model,
        "source_format": context.source_format,
        "tenant_id": context.tenant_id,
        "request_id": context.request_id,
        "conversation_fingerprint": context.conversation_fingerprint,
        "reasoning_effort": context.reasoning_effort,
        "stream": context.stream,
        "batch_index": context.batch_index,
        "direct_passthrough": true,
        "usage": usage_telemetry_json(usage),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens),
        "provider_call_count": 1,
        "task_count": 0,
        "child_agent_count": 0,
        "tool_call_count": 0,
        "verification": null,
        "context": null,
    })
}

fn stream_provider_telemetry_record(
    context: &TelemetryContext,
    usage: &ProviderUsage,
    finish_reason: Option<&str>,
    error_type: Option<&str>,
    saw_tool_delta: bool,
) -> serde_json::Value {
    serde_json::json!({
        "event": "api_usage",
        "event_version": 1,
        "timestamp_ms": telemetry_timestamp_ms(),
        "route": context.route,
        "model": context.model,
        "source_format": context.source_format,
        "tenant_id": context.tenant_id,
        "request_id": context.request_id,
        "conversation_fingerprint": context.conversation_fingerprint,
        "reasoning_effort": context.reasoning_effort,
        "stream": true,
        "batch_index": context.batch_index,
        "direct_passthrough": false,
        "usage": usage_telemetry_json(usage),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens),
        "provider_call_count": 1,
        "task_count": 1,
        "child_agent_count": 0,
        "tool_delta_seen": saw_tool_delta,
        "finish_reason": finish_reason,
        "error_type": error_type,
        "verification": null,
        "context": null,
    })
}

fn usage_telemetry_json(usage: &ProviderUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens),
    })
}

fn telemetry_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or_default()
}

fn response_usage(value: &serde_json::Value) -> ProviderUsage {
    value
        .get("usage")
        .and_then(|usage| serde_json::from_value::<ProviderUsage>(usage.clone()).ok())
        .unwrap_or_default()
}

fn public_reasoning_summary(output: &KernelOutput) -> String {
    let mut lines = vec!["Multi-agent process summary:".to_string()];
    let mut saw_spawn = false;

    for event in &output.trace_events {
        if let KernelTraceEvent::SpawnPlan {
            reason, children, ..
        } = event
        {
            saw_spawn = true;
            let objectives = children
                .iter()
                .take(4)
                .map(|child| {
                    format!(
                        "{}={}",
                        child.task_id.as_ref(),
                        truncate_summary_text(&child.objective, 96)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let suffix = if children.len() > 4 {
                format!(", plus {} more", children.len() - 4)
            } else {
                String::new()
            };
            lines.push(format!(
                "- Orchestrator scheduled {} bounded child agent(s) for {}{}. Reason: {}.",
                children.len(),
                if objectives.is_empty() {
                    "no listed objectives".to_string()
                } else {
                    objectives
                },
                suffix,
                truncate_summary_text(reason, 160)
            ));
        }
    }

    if !saw_spawn {
        lines.push(
            "- No child agents were spawned; the leader handled the request within bounded orchestration."
                .to_string(),
        );
    }

    let agent_output_summaries = summary_agent_output_lines(output).unwrap_or_else(|| {
        output
            .trace_events
            .iter()
            .filter_map(agent_output_summary_line)
            .collect::<Vec<_>>()
    });
    if !agent_output_summaries.is_empty() {
        lines.push("- Agent output summaries:".to_string());
        lines.extend(agent_output_summaries);
    }

    for event in &output.trace_events {
        let KernelTraceEvent::AgentOutput {
            text_outputs,
            tool_calls,
            ..
        } = event
        else {
            continue;
        };
        if text_outputs.is_empty() && !tool_calls.is_empty() {
            lines.push(format!(
                "- {} internal tool call(s) were recorded without exposing child tool payloads.",
                tool_calls.len()
            ));
        }
    }

    if output.tool_calls.is_empty() {
        lines.push("- No main-agent tool call needs to be surfaced to the client.".to_string());
    } else {
        let names = output
            .tool_calls
            .iter()
            .take(4)
            .map(|call| call.tool_name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!(
            "- Main agent surfaced {} tool call(s) to the client: {}.",
            output.tool_calls.len(),
            names
        ));
    }

    let verification = &output.verification;
    lines.push(format!(
        "- Verification {} with {} issue(s), {} unresolved tool call(s), {} provider call(s), and {} orchestration tokens recorded against accounting reference {}.",
        if verification.passed { "passed" } else { "requires client tool action" },
        verification.issues.len(),
        verification.unresolved_tool_calls.len(),
        output.provider_call_count,
        verification.budget_summary.token_used,
        verification.budget_summary.token_budget
    ));

    if !output.encrypted_subagent_state.is_empty() {
        lines.push(
            "- Raw sub-agent state remains encrypted; public reasoning contains deterministic summaries of text artifacts only."
                .to_string(),
        );
    }

    lines.join("\n")
}

fn agent_output_summary_line(event: &KernelTraceEvent) -> Option<String> {
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

    let summary = summarize_text_outputs(text_outputs);

    Some(format!(
        "  - {} ({}): {}",
        task_id.as_ref(),
        agent_role_label(role),
        summary
    ))
}

fn agent_role_label(role: &AgentRole) -> &'static str {
    match role {
        AgentRole::Leader => "Leader agent",
        AgentRole::Worker => "Worker agent",
        AgentRole::Verifier => "Verifier agent",
        AgentRole::ReasoningSummarizer => "Reasoning summary agent",
        AgentRole::Synthesizer => "Synthesizer agent",
    }
}

fn summary_agent_output_lines(output: &KernelOutput) -> Option<Vec<String>> {
    let summary_text = output.trace_events.iter().find_map(|event| {
        let KernelTraceEvent::AgentOutput {
            task_id,
            role,
            text_outputs,
            ..
        } = event
        else {
            return None;
        };
        if task_id.as_ref() != "reasoning-summary" || *role != AgentRole::ReasoningSummarizer {
            return None;
        }
        let text = text_outputs
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        (!text.trim().is_empty()).then_some(text)
    })?;

    let lines = summary_text
        .lines()
        .map(clean_summary_agent_line)
        .filter(|line| !line.is_empty())
        .map(|line| format!("  - {line}"))
        .collect::<Vec<_>>();
    (!lines.is_empty()).then_some(lines)
}

fn clean_summary_agent_line(line: &str) -> String {
    let line = line.trim().trim_start_matches(['-', '*', ' ', '\t']).trim();
    if line.eq_ignore_ascii_case("multi-agent process summary:")
        || line.eq_ignore_ascii_case("agent output summaries:")
    {
        String::new()
    } else {
        line.to_string()
    }
}

fn summarize_text_outputs(text_outputs: &[String]) -> String {
    let joined = text_outputs
        .iter()
        .map(|text| summarize_text_output(text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");

    if joined.is_empty() {
        "empty text artifact".to_string()
    } else {
        truncate_summary_text(&joined, 280)
    }
}

fn summarize_text_output(text: &str) -> String {
    let compact = compact_summary_whitespace(text);
    if compact.is_empty() {
        return compact;
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&compact) {
        let summary = summarize_json_value(&value);
        if !summary.is_empty() {
            return truncate_summary_text(&summary, 280);
        }
    }

    if let Some(summary) = summarize_json_like_fragment(&compact) {
        return truncate_summary_text(&summary, 280);
    }

    truncate_summary_text(&compact, 280)
}

fn compact_summary_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn summarize_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => compact_summary_whitespace(text),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => value.to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(json_value_scalar_summary)
            .take(4)
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Object(object) => ordered_summary_entries(object)
            .into_iter()
            .filter_map(|(key, value)| json_entry_summary(key, value))
            .take(6)
            .collect::<Vec<_>>()
            .join("; "),
        serde_json::Value::Null => String::new(),
    }
}

fn ordered_summary_entries(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Vec<(&String, &serde_json::Value)> {
    let mut entries = object.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(key, _)| summary_key_rank(key));
    entries
}

fn summary_key_rank(key: &str) -> usize {
    match key.to_ascii_lowercase().as_str() {
        "summary" => 0,
        "answer" => 1,
        "conclusion" => 2,
        "result" => 3,
        "intent" => 4,
        "observation" => 5,
        _ => 16,
    }
}

fn json_entry_summary(key: &str, value: &serde_json::Value) -> Option<String> {
    let value_summary = match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(json_value_scalar_summary)
            .take(3)
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Object(_) => summarize_json_value(value),
        _ => json_value_scalar_summary(value).unwrap_or_default(),
    };

    if value_summary.is_empty() {
        None
    } else {
        Some(format!("{key}: {value_summary}"))
    }
}

fn json_value_scalar_summary(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            let compact = compact_summary_whitespace(text);
            (!compact.is_empty()).then_some(compact)
        }
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => Some(value.to_string()),
        _ => None,
    }
}

fn summarize_json_like_fragment(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return None;
    }

    let chars = trimmed.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut pairs = Vec::new();

    while index < chars.len() && pairs.len() < 6 {
        let Some(key_start) = chars[index..]
            .iter()
            .position(|ch| *ch == '"')
            .map(|offset| index + offset)
        else {
            break;
        };
        let Some((key, after_key)) = parse_quoted_fragment(&chars, key_start) else {
            break;
        };
        index = skip_summary_whitespace(&chars, after_key);
        if chars.get(index) != Some(&':') {
            continue;
        }
        index = skip_summary_whitespace(&chars, index + 1);

        let Some((value, after_value)) = parse_json_like_value(&chars, index) else {
            break;
        };
        index = after_value;
        let compact_value = compact_summary_whitespace(&value);
        if !key.is_empty() && !compact_value.is_empty() {
            pairs.push(format!("{key}: {compact_value}"));
        }
    }

    (!pairs.is_empty()).then(|| pairs.join("; "))
}

fn parse_json_like_value(chars: &[char], index: usize) -> Option<(String, usize)> {
    match chars.get(index) {
        Some('"') => parse_quoted_fragment(chars, index),
        Some('{') | Some('[') => None,
        Some(_) => {
            let mut end = index;
            while end < chars.len() && !matches!(chars[end], ',' | '}' | ']') {
                end += 1;
            }
            let value = chars[index..end]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            (!value.is_empty()).then_some((value, end))
        }
        None => None,
    }
}

fn parse_quoted_fragment(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars.get(start) != Some(&'"') {
        return None;
    }

    let mut output = String::new();
    let mut escaped = false;
    for (offset, ch) in chars.iter().enumerate().skip(start + 1) {
        if escaped {
            output.push(*ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some((output, offset + 1)),
            _ => output.push(*ch),
        }
    }
    None
}

fn skip_summary_whitespace(chars: &[char], mut index: usize) -> usize {
    while chars.get(index).is_some_and(|ch| ch.is_whitespace()) {
        index += 1;
    }
    index
}

fn truncate_summary_text(text: &str, max_chars: usize) -> String {
    let mut output = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        output.push_str("...");
    }
    output
}

fn text_with_public_reasoning(
    final_text: &str,
    summary: &str,
    thinking_format: &ThinkingFormat,
) -> String {
    let summary = summary.trim();
    match thinking_format {
        ThinkingFormat::GemmaSystemToken => {
            format!("<|channel>thought\n{summary}\n<channel|>\n{final_text}")
        }
        ThinkingFormat::QwenChatTemplate | ThinkingFormat::QwenDashScope | ThinkingFormat::Auto => {
            format!("<think>\n{summary}\n</think>\n\n{final_text}")
        }
    }
}

fn format_openai_response(
    model: String,
    output: KernelOutput,
    include_encrypted_state: bool,
    tool_response_format: OpenAiToolResponseFormat,
    include_public_reasoning: bool,
) -> serde_json::Value {
    let encrypted_state = output.encrypted_subagent_state.clone();
    let usage = openai_usage_json(&output.usage);
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));
    let mut value = if output.verification.passed {
        serde_json::json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.final_text
                },
                "finish_reason": "stop"
            }],
            "usage": usage
        })
    } else if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
        let function_call = output
            .tool_calls
            .into_iter()
            .next()
            .map(openai_legacy_function_call_json)
            .unwrap_or_else(|| serde_json::json!({"name": "unknown", "arguments": "{}"}));
        serde_json::json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "function_call": function_call
                },
                "finish_reason": "function_call"
            }],
            "usage": usage
        })
    } else {
        serde_json::json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": output.tool_calls.into_iter().map(openai_tool_call_json).collect::<Vec<_>>()
                },
                "finish_reason": "tool_calls"
            }],
            "usage": usage
        })
    };

    if include_encrypted_state {
        value["encrypted_agent_state"] = serde_json::json!(encrypted_state);
    }

    if let Some(summary) = reasoning_summary
        && let Some(message) = value
            .get_mut("choices")
            .and_then(|choices| choices.get_mut(0))
            .and_then(|choice| choice.get_mut("message"))
            .and_then(|message| message.as_object_mut())
    {
        message.insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(summary.clone()),
        );
        message.insert(
            "reasoning".to_string(),
            serde_json::json!({"summary": summary}),
        );
    }

    value
}

fn openai_usage_json(usage: &provider_core::ProviderUsage) -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens)
    })
}

fn format_openai_completion_response(
    model: String,
    output: KernelOutput,
    include_public_reasoning: bool,
    thinking_format: ThinkingFormat,
) -> serde_json::Value {
    let text = if include_public_reasoning {
        text_with_public_reasoning(
            &output.final_text,
            &public_reasoning_summary(&output),
            &thinking_format,
        )
    } else {
        output.final_text.clone()
    };
    serde_json::json!({
        "id": format!("cmpl-{}", Uuid::new_v4()),
        "object": "text_completion",
        "created": 0,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": openai_usage_json(&output.usage)
    })
}

fn format_openai_completion_stream_response(
    model: String,
    output: KernelOutput,
    include_public_reasoning: bool,
    thinking_format: ThinkingFormat,
) -> Response {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let text = if include_public_reasoning {
        text_with_public_reasoning(
            &output.final_text,
            &public_reasoning_summary(&output),
            &thinking_format,
        )
    } else {
        output.final_text.clone()
    };
    let mut body = String::new();
    body.push_str(&sse_data(openai_completion_stream_chunk(
        &id,
        &model,
        &text,
        serde_json::Value::Null,
    )));
    body.push_str(&sse_data(openai_completion_stream_chunk(
        &id,
        &model,
        "",
        serde_json::Value::String("stop".to_string()),
    )));
    body.push_str("data: [DONE]\n\n");
    sse_response(body)
}

fn format_openai_stream_response(
    model: String,
    output: KernelOutput,
    tool_response_format: OpenAiToolResponseFormat,
    include_public_reasoning: bool,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let mut body = String::new();
    body.push_str(&openai_stream_role_chunk(&id, &model));
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));

    if output.verification.passed {
        if let Some(summary) = reasoning_summary {
            body.push_str(&sse_data(openai_stream_chunk(
                &id,
                &model,
                serde_json::json!({
                    "reasoning_content": summary.clone(),
                    "reasoning": {"summary": summary}
                }),
                serde_json::Value::Null,
            )));
        }
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"content": output.final_text}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "stop"));
    } else if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
        if let Some(summary) = reasoning_summary {
            body.push_str(&sse_data(openai_stream_chunk(
                &id,
                &model,
                serde_json::json!({
                    "reasoning_content": summary.clone(),
                    "reasoning": {"summary": summary}
                }),
                serde_json::Value::Null,
            )));
        }
        let function_call = output
            .tool_calls
            .into_iter()
            .next()
            .map(openai_legacy_function_call_json)
            .unwrap_or_else(|| serde_json::json!({"name": "unknown", "arguments": "{}"}));
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"function_call": function_call}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "function_call"));
    } else {
        if let Some(summary) = reasoning_summary {
            body.push_str(&sse_data(openai_stream_chunk(
                &id,
                &model,
                serde_json::json!({
                    "reasoning_content": summary.clone(),
                    "reasoning": {"summary": summary}
                }),
                serde_json::Value::Null,
            )));
        }
        let tool_calls = output
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| openai_stream_tool_call_delta(index, call))
            .collect::<Vec<_>>();
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"tool_calls": tool_calls}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "tool_calls"));
    }

    body.push_str("data: [DONE]\n\n");
    sse_response(body)
}

fn openai_stream_response_from_completion(value: serde_json::Value) -> Response {
    let id = value
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("chatcmpl-{}", Uuid::new_v4()));
    let model = value
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("mock")
        .to_string();
    let message = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .cloned()
        .unwrap_or_default();

    let mut body = String::new();
    body.push_str(&openai_stream_role_chunk(&id, &model));
    if let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
        let tool_calls = tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let mut call = call.clone();
                call["index"] = serde_json::json!(index);
                call
            })
            .collect::<Vec<_>>();
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"tool_calls": tool_calls}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "tool_calls"));
    } else if let Some(function_call) = message.get("function_call") {
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"function_call": function_call}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "function_call"));
    } else {
        let content = message
            .get("content")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"content": content}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "stop"));
    }
    body.push_str("data: [DONE]\n\n");
    sse_response(body)
}

fn openai_stream_role_chunk(id: &str, model: &str) -> String {
    sse_data(openai_stream_chunk(
        id,
        model,
        serde_json::json!({"role": "assistant"}),
        serde_json::Value::Null,
    ))
}

fn openai_stream_finish_chunk(id: &str, model: &str, finish_reason: &str) -> String {
    sse_data(openai_stream_chunk(
        id,
        model,
        serde_json::json!({}),
        serde_json::Value::String(finish_reason.to_string()),
    ))
}

fn openai_stream_chunk(
    id: &str,
    model: &str,
    delta: serde_json::Value,
    finish_reason: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

fn openai_completion_stream_chunk(
    id: &str,
    model: &str,
    text: &str,
    finish_reason: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "text_completion",
        "created": 0,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": finish_reason
        }]
    })
}

fn openai_stream_tool_call_delta(index: usize, call: ToolCallRecord) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "id": call.tool_call_id.as_ref(),
        "type": "function",
        "function": {
            "name": call.tool_name,
            "arguments": call.arguments_json.to_string()
        }
    })
}

fn format_openai_completion_provider_stream_response(
    model: String,
    provider_stream: ProviderStream,
    telemetry_context: TelemetryContext,
) -> Response {
    let state = OpenAiCompletionStreamState {
        id: format!("cmpl-{}", Uuid::new_v4()),
        model,
        provider_stream,
        telemetry_context,
        usage: ProviderUsage::default(),
        pending: VecDeque::new(),
        finished: false,
        telemetry_emitted: false,
    };

    sse_stream_response(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(chunk) = state.pending.pop_front() {
                return Some((Ok(Bytes::from(chunk)), state));
            }

            if state.finished {
                return None;
            }

            match state.provider_stream.next().await {
                Some(Ok(event)) => state.push_openai_completion_event(event),
                Some(Err(error)) => {
                    state.pending.push_back(sse_data(serde_json::json!({
                        "error": {
                            "message": error.to_string(),
                            "type": "provider_stream_error"
                        }
                    })));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
                    state.emit_stream_telemetry(None, Some("provider_stream_error"));
                }
                None => {
                    state
                        .pending
                        .push_back(sse_data(openai_completion_stream_chunk(
                            &state.id,
                            &state.model,
                            "",
                            serde_json::Value::String("stop".to_string()),
                        )));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
                    state.emit_stream_telemetry(Some("stop"), None);
                }
            }
        }
    }))
}

struct OpenAiCompletionStreamState {
    id: String,
    model: String,
    provider_stream: ProviderStream,
    telemetry_context: TelemetryContext,
    usage: ProviderUsage,
    pending: VecDeque<String>,
    finished: bool,
    telemetry_emitted: bool,
}

impl OpenAiCompletionStreamState {
    fn push_openai_completion_event(&mut self, event: ProviderStreamEvent) {
        match event {
            ProviderStreamEvent::TextDelta { text } if !text.is_empty() => {
                self.pending
                    .push_back(sse_data(openai_completion_stream_chunk(
                        &self.id,
                        &self.model,
                        &text,
                        serde_json::Value::Null,
                    )));
            }
            ProviderStreamEvent::Finish { reason } => {
                let finish_reason = openai_completion_finish_reason(&reason);
                self.pending
                    .push_back(sse_data(openai_completion_stream_chunk(
                        &self.id,
                        &self.model,
                        "",
                        serde_json::Value::String(finish_reason.to_string()),
                    )));
                self.pending.push_back("data: [DONE]\n\n".to_string());
                self.finished = true;
                self.emit_stream_telemetry(Some(finish_reason), None);
            }
            ProviderStreamEvent::Usage { usage } => {
                self.usage = usage;
            }
            ProviderStreamEvent::ToolCallDelta { .. } | ProviderStreamEvent::TextDelta { .. } => {}
        }
    }

    fn emit_stream_telemetry(&mut self, finish_reason: Option<&str>, error_type: Option<&str>) {
        if self.telemetry_emitted {
            return;
        }
        self.telemetry_emitted = true;
        emit_telemetry(stream_provider_telemetry_record(
            &self.telemetry_context,
            &self.usage,
            finish_reason,
            error_type,
            false,
        ));
    }
}

fn openai_completion_finish_reason(reason: &ProviderFinishReason) -> &'static str {
    match reason {
        ProviderFinishReason::Length => "length",
        ProviderFinishReason::Other(_) => "stop",
        _ => "stop",
    }
}

fn format_openai_provider_stream_response(
    model: String,
    provider_stream: ProviderStream,
    tool_response_format: OpenAiToolResponseFormat,
    telemetry_context: TelemetryContext,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let state = OpenAiStreamState {
        id,
        model,
        provider_stream,
        tool_response_format,
        telemetry_context,
        usage: ProviderUsage::default(),
        pending: VecDeque::from([openai_stream_role_chunk_placeholder()]),
        finished: false,
        saw_tool_delta: false,
        telemetry_emitted: false,
    };

    sse_stream_response(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(chunk) = state.pending.pop_front() {
                let chunk = if chunk == OPENAI_ROLE_CHUNK_PLACEHOLDER {
                    openai_stream_role_chunk(&state.id, &state.model)
                } else {
                    chunk
                };
                return Some((Ok(Bytes::from(chunk)), state));
            }

            if state.finished {
                return None;
            }

            match state.provider_stream.next().await {
                Some(Ok(event)) => state.push_openai_event(event),
                Some(Err(error)) => {
                    state.pending.push_back(sse_data(serde_json::json!({
                        "error": {
                            "message": error.to_string(),
                            "type": "provider_stream_error"
                        }
                    })));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
                    state.emit_stream_telemetry(None, Some("provider_stream_error"));
                }
                None => {
                    state.pending.push_back(openai_stream_finish_chunk(
                        &state.id,
                        &state.model,
                        "stop",
                    ));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
                    state.emit_stream_telemetry(Some("stop"), None);
                }
            }
        }
    }))
}

const OPENAI_ROLE_CHUNK_PLACEHOLDER: &str = "__openai_role_chunk__";

fn openai_stream_role_chunk_placeholder() -> String {
    OPENAI_ROLE_CHUNK_PLACEHOLDER.to_string()
}

struct OpenAiStreamState {
    id: String,
    model: String,
    provider_stream: ProviderStream,
    tool_response_format: OpenAiToolResponseFormat,
    telemetry_context: TelemetryContext,
    usage: ProviderUsage,
    pending: VecDeque<String>,
    finished: bool,
    saw_tool_delta: bool,
    telemetry_emitted: bool,
}

impl OpenAiStreamState {
    fn push_openai_event(&mut self, event: ProviderStreamEvent) {
        match event {
            ProviderStreamEvent::TextDelta { text } if !text.is_empty() => {
                let id = &self.id;
                let model = &self.model;
                self.pending.push_back(sse_data(openai_stream_chunk(
                    id,
                    model,
                    serde_json::json!({"content": text}),
                    serde_json::Value::Null,
                )));
            }
            ProviderStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                self.saw_tool_delta = true;
                let delta =
                    if self.tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
                        openai_legacy_function_call_delta(name, arguments_delta)
                    } else {
                        serde_json::json!({
                            "tool_calls": [openai_provider_tool_call_delta(
                                index,
                                id,
                                name,
                                arguments_delta
                            )]
                        })
                    };
                self.pending.push_back(sse_data(openai_stream_chunk(
                    &self.id,
                    &self.model,
                    delta,
                    serde_json::Value::Null,
                )));
            }
            ProviderStreamEvent::Finish { reason } => {
                let finish_reason =
                    openai_finish_reason(&reason, self.tool_response_format, self.saw_tool_delta);
                self.pending.push_back(openai_stream_finish_chunk(
                    &self.id,
                    &self.model,
                    finish_reason,
                ));
                self.pending.push_back("data: [DONE]\n\n".to_string());
                self.finished = true;
                self.emit_stream_telemetry(Some(finish_reason), None);
            }
            ProviderStreamEvent::Usage { usage } => {
                self.usage = usage;
            }
            ProviderStreamEvent::TextDelta { .. } => {}
        }
    }

    fn emit_stream_telemetry(&mut self, finish_reason: Option<&str>, error_type: Option<&str>) {
        if self.telemetry_emitted {
            return;
        }
        self.telemetry_emitted = true;
        emit_telemetry(stream_provider_telemetry_record(
            &self.telemetry_context,
            &self.usage,
            finish_reason,
            error_type,
            self.saw_tool_delta,
        ));
    }
}

fn openai_provider_tool_call_delta(
    index: usize,
    id: Option<ToolCallId>,
    name: Option<String>,
    arguments_delta: String,
) -> serde_json::Value {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), serde_json::Value::String(name));
    }
    if !arguments_delta.is_empty() {
        function.insert(
            "arguments".to_string(),
            serde_json::Value::String(arguments_delta),
        );
    }

    let mut value = serde_json::Map::new();
    value.insert("index".to_string(), serde_json::json!(index));
    if let Some(id) = id {
        value.insert(
            "id".to_string(),
            serde_json::Value::String(id.as_ref().to_string()),
        );
        value.insert(
            "type".to_string(),
            serde_json::Value::String("function".to_string()),
        );
    }
    value.insert("function".to_string(), serde_json::Value::Object(function));
    serde_json::Value::Object(value)
}

fn openai_legacy_function_call_delta(
    name: Option<String>,
    arguments_delta: String,
) -> serde_json::Value {
    let mut function_call = serde_json::Map::new();
    if let Some(name) = name {
        function_call.insert("name".to_string(), serde_json::Value::String(name));
    }
    if !arguments_delta.is_empty() {
        function_call.insert(
            "arguments".to_string(),
            serde_json::Value::String(arguments_delta),
        );
    }
    serde_json::json!({"function_call": function_call})
}

fn openai_finish_reason(
    reason: &ProviderFinishReason,
    tool_response_format: OpenAiToolResponseFormat,
    saw_tool_delta: bool,
) -> &'static str {
    if saw_tool_delta && matches!(reason, ProviderFinishReason::Stop) {
        return if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
            "function_call"
        } else {
            "tool_calls"
        };
    }

    match reason {
        ProviderFinishReason::Stop => "stop",
        ProviderFinishReason::ToolCalls
            if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions =>
        {
            "function_call"
        }
        ProviderFinishReason::ToolCalls => "tool_calls",
        ProviderFinishReason::FunctionCall => "function_call",
        ProviderFinishReason::Length => "length",
        ProviderFinishReason::Other(_) => "stop",
    }
}

fn include_encrypted_subagent_state(metadata: &serde_json::Value) -> bool {
    metadata
        .get("include_encrypted_subagent_state")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn openai_public_reasoning_requested(request: &OpenAiChatRequest) -> bool {
    public_reasoning_requested_from_parts(
        &request.extra,
        &request.metadata,
        &request.reasoning,
        &request.thinking,
    )
}

fn anthropic_public_reasoning_requested(request: &AnthropicMessagesRequest) -> bool {
    public_reasoning_requested_from_parts(
        &request.extra,
        &request.metadata,
        &request.reasoning,
        &request.thinking,
    )
}

fn public_reasoning_requested_from_parts(
    extra: &BTreeMap<String, serde_json::Value>,
    metadata: &serde_json::Value,
    reasoning: &serde_json::Value,
    thinking: &serde_json::Value,
) -> bool {
    extra
        .get("include_reasoning")
        .or_else(|| extra.get("include_thinking"))
        .or_else(|| extra.get("show_reasoning"))
        .or_else(|| extra.get("return_reasoning"))
        .is_some_and(bool_like_true)
        || metadata
            .get("include_reasoning")
            .or_else(|| metadata.get("include_thinking"))
            .or_else(|| metadata.get("show_reasoning"))
            .or_else(|| metadata.get("return_reasoning"))
            .is_some_and(bool_like_true)
        || reasoning_summary_requested(reasoning)
        || thinking_summary_requested(thinking)
}

fn bool_like_true(value: &serde_json::Value) -> bool {
    if value.as_bool() == Some(true) {
        return true;
    }
    value
        .as_str()
        .map(|text| {
            matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "true" | "1" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn reasoning_summary_requested(reasoning: &serde_json::Value) -> bool {
    reasoning.get("summary").is_some_and(|value| {
        value.as_bool() == Some(true)
            || value.as_str().is_some_and(|text| {
                !matches!(
                    text.trim().to_ascii_lowercase().as_str(),
                    "" | "false" | "off" | "none" | "disabled"
                )
            })
    })
}

fn thinking_summary_requested(thinking: &serde_json::Value) -> bool {
    thinking.get("enabled").is_some_and(bool_like_true)
        || thinking
            .get("type")
            .and_then(|value| value.as_str())
            .is_some_and(|kind| kind.eq_ignore_ascii_case("enabled"))
}

fn requires_full_orchestration_before_stream(
    request: &NormalizedRequest,
    include_public_reasoning: bool,
) -> bool {
    if include_public_reasoning
        || include_encrypted_subagent_state(&request.metadata)
        || matches!(
            request.reasoning_effort,
            ReasoningEffort::High | ReasoningEffort::XHigh
        )
    {
        return true;
    }

    last_user_text(request)
        .map(|text| text.to_ascii_lowercase().contains("spawn"))
        .unwrap_or(false)
}

fn openai_tool_call_json(call: ToolCallRecord) -> serde_json::Value {
    serde_json::json!({
        "id": call.tool_call_id.as_ref(),
        "type": "function",
        "function": {
            "name": call.tool_name,
            "arguments": call.arguments_json.to_string()
        }
    })
}

fn openai_legacy_function_call_json(call: ToolCallRecord) -> serde_json::Value {
    serde_json::json!({
        "name": call.tool_name,
        "arguments": call.arguments_json.to_string()
    })
}

fn format_anthropic_response(
    model: String,
    output: KernelOutput,
    include_encrypted_state: bool,
    include_public_reasoning: bool,
) -> serde_json::Value {
    let encrypted_state = output.encrypted_subagent_state.clone();
    let usage = anthropic_usage_json(&output.usage);
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));
    let mut content = Vec::new();
    if let Some(summary) = reasoning_summary {
        content.push(serde_json::json!({
            "type": "thinking",
            "thinking": summary,
            "signature": ""
        }));
    }
    let stop_reason = if output.verification.passed {
        content.push(serde_json::json!({
            "type": "text",
            "text": output.final_text
        }));
        "end_turn"
    } else {
        content.extend(output.tool_calls.into_iter().map(anthropic_tool_use_json));
        "tool_use"
    };

    let mut value = serde_json::json!({
        "id": format!("msg_{}", Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage
    });

    if include_encrypted_state {
        value["encrypted_agent_state"] = serde_json::json!(encrypted_state);
    }

    value
}

fn anthropic_usage_json(usage: &provider_core::ProviderUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens
    })
}

fn format_anthropic_stream_response(
    model: String,
    output: KernelOutput,
    include_public_reasoning: bool,
) -> Response {
    let id = format!("msg_{}", Uuid::new_v4());
    let mut body = String::new();
    body.push_str(&anthropic_message_start_event(&id, &model));
    let mut index = 0_usize;

    if include_public_reasoning {
        body.push_str(&anthropic_thinking_events(
            index,
            &public_reasoning_summary(&output),
        ));
        index += 1;
    }

    if output.verification.passed {
        body.push_str(&sse_event(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
        ));
        body.push_str(&sse_event(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": output.final_text}
            }),
        ));
        body.push_str(&anthropic_content_block_stop_event(index));
        body.push_str(&anthropic_message_delta_event("end_turn"));
    } else {
        for (offset, call) in output.tool_calls.into_iter().enumerate() {
            body.push_str(&anthropic_tool_use_events(index + offset, call));
        }
        body.push_str(&anthropic_message_delta_event("tool_use"));
    }

    body.push_str(&sse_event(
        "message_stop",
        serde_json::json!({"type": "message_stop"}),
    ));
    sse_response(body)
}

fn anthropic_thinking_events(index: usize, summary: &str) -> String {
    let mut body = String::new();
    body.push_str(&sse_event(
        "content_block_start",
        serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        }),
    ));
    body.push_str(&sse_event(
        "content_block_delta",
        serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": summary}
        }),
    ));
    body.push_str(&anthropic_content_block_stop_event(index));
    body
}

fn anthropic_stream_response_from_message(value: serde_json::Value) -> Response {
    let id = value
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("msg_{}", Uuid::new_v4()));
    let model = value
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("mock")
        .to_string();
    let mut body = String::new();
    body.push_str(&anthropic_message_start_event(&id, &model));

    let content = value
        .get("content")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut stop_reason = "end_turn";
    for (index, block) in content.into_iter().enumerate() {
        match block.get("type").and_then(|value| value.as_str()) {
            Some("tool_use") => {
                stop_reason = "tool_use";
                body.push_str(&sse_event(
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id").cloned().unwrap_or_default(),
                            "name": block.get("name").cloned().unwrap_or_default(),
                            "input": {}
                        }
                    }),
                ));
                body.push_str(&sse_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": block.get("input").cloned().unwrap_or_default().to_string()
                        }
                    }),
                ));
                body.push_str(&anthropic_content_block_stop_event(index));
            }
            Some("thinking") => {
                let thinking = block
                    .get("thinking")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                body.push_str(&anthropic_thinking_events(index, thinking));
            }
            _ => {
                let text = block
                    .get("text")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                body.push_str(&sse_event(
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""}
                    }),
                ));
                body.push_str(&sse_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text}
                    }),
                ));
                body.push_str(&anthropic_content_block_stop_event(index));
            }
        }
    }
    body.push_str(&anthropic_message_delta_event(stop_reason));
    body.push_str(&sse_event(
        "message_stop",
        serde_json::json!({"type": "message_stop"}),
    ));
    sse_response(body)
}

fn anthropic_message_start_event(id: &str, model: &str) -> String {
    sse_event(
        "message_start",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }),
    )
}

fn anthropic_tool_use_events(index: usize, call: ToolCallRecord) -> String {
    let mut body = String::new();
    body.push_str(&sse_event(
        "content_block_start",
        serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": call.tool_call_id.as_ref(),
                "name": call.tool_name,
                "input": {}
            }
        }),
    ));
    body.push_str(&sse_event(
        "content_block_delta",
        serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": call.arguments_json.to_string()
            }
        }),
    ));
    body.push_str(&anthropic_content_block_stop_event(index));
    body
}

fn anthropic_content_block_stop_event(index: usize) -> String {
    sse_event(
        "content_block_stop",
        serde_json::json!({
            "type": "content_block_stop",
            "index": index
        }),
    )
}

fn anthropic_message_delta_event(stop_reason: &str) -> String {
    sse_event(
        "message_delta",
        serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": {"output_tokens": 0}
        }),
    )
}

fn format_anthropic_provider_stream_response(
    model: String,
    provider_stream: ProviderStream,
    telemetry_context: TelemetryContext,
) -> Response {
    let id = format!("msg_{}", Uuid::new_v4());
    let state = AnthropicStreamState {
        id,
        model,
        provider_stream,
        telemetry_context,
        usage: ProviderUsage::default(),
        pending: VecDeque::from([anthropic_message_start_placeholder()]),
        text_block_open: false,
        tool_blocks_open: BTreeSet::new(),
        finished: false,
        saw_tool_delta: false,
        telemetry_emitted: false,
    };

    sse_stream_response(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(chunk) = state.pending.pop_front() {
                let chunk = if chunk == ANTHROPIC_MESSAGE_START_PLACEHOLDER {
                    anthropic_message_start_event(&state.id, &state.model)
                } else {
                    chunk
                };
                return Some((Ok(Bytes::from(chunk)), state));
            }

            if state.finished {
                return None;
            }

            match state.provider_stream.next().await {
                Some(Ok(event)) => state.push_anthropic_event(event),
                Some(Err(error)) => {
                    state.close_open_blocks();
                    state.pending.push_back(sse_event(
                        "error",
                        serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "provider_stream_error",
                                "message": error.to_string()
                            }
                        }),
                    ));
                    state.pending.push_back(sse_event(
                        "message_stop",
                        serde_json::json!({"type": "message_stop"}),
                    ));
                    state.finished = true;
                    state.emit_stream_telemetry(None, Some("provider_stream_error"));
                }
                None => {
                    state.finish_with_reason("end_turn");
                }
            }
        }
    }))
}

const ANTHROPIC_MESSAGE_START_PLACEHOLDER: &str = "__anthropic_message_start__";

fn anthropic_message_start_placeholder() -> String {
    ANTHROPIC_MESSAGE_START_PLACEHOLDER.to_string()
}

struct AnthropicStreamState {
    id: String,
    model: String,
    provider_stream: ProviderStream,
    telemetry_context: TelemetryContext,
    usage: ProviderUsage,
    pending: VecDeque<String>,
    text_block_open: bool,
    tool_blocks_open: BTreeSet<usize>,
    finished: bool,
    saw_tool_delta: bool,
    telemetry_emitted: bool,
}

impl AnthropicStreamState {
    fn push_anthropic_event(&mut self, event: ProviderStreamEvent) {
        match event {
            ProviderStreamEvent::TextDelta { text } if !text.is_empty() => {
                if !self.text_block_open {
                    self.pending.push_back(sse_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""}
                        }),
                    ));
                    self.text_block_open = true;
                }
                self.pending.push_back(sse_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text}
                    }),
                ));
            }
            ProviderStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                self.saw_tool_delta = true;
                if self.tool_blocks_open.insert(index) {
                    self.pending.push_back(sse_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": id.map(|id| id.as_ref().to_string()).unwrap_or_else(|| format!("toolu_{index}")),
                                "name": name.unwrap_or_else(|| "unknown".to_string()),
                                "input": {}
                            }
                        }),
                    ));
                }
                if !arguments_delta.is_empty() {
                    self.pending.push_back(sse_event(
                        "content_block_delta",
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments_delta
                            }
                        }),
                    ));
                }
            }
            ProviderStreamEvent::Finish { reason } => {
                let stop_reason = anthropic_stop_reason(&reason, self.saw_tool_delta);
                self.finish_with_reason(stop_reason);
            }
            ProviderStreamEvent::Usage { usage } => {
                self.usage = usage;
            }
            ProviderStreamEvent::TextDelta { .. } => {}
        }
    }

    fn close_open_blocks(&mut self) {
        if self.text_block_open {
            self.pending
                .push_back(anthropic_content_block_stop_event(0));
            self.text_block_open = false;
        }
        let tool_indices = std::mem::take(&mut self.tool_blocks_open);
        for index in tool_indices {
            self.pending
                .push_back(anthropic_content_block_stop_event(index));
        }
    }

    fn finish_with_reason(&mut self, stop_reason: &str) {
        self.close_open_blocks();
        self.pending
            .push_back(anthropic_message_delta_event(stop_reason));
        self.pending.push_back(sse_event(
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        ));
        self.finished = true;
        self.emit_stream_telemetry(Some(stop_reason), None);
    }

    fn emit_stream_telemetry(&mut self, finish_reason: Option<&str>, error_type: Option<&str>) {
        if self.telemetry_emitted {
            return;
        }
        self.telemetry_emitted = true;
        emit_telemetry(stream_provider_telemetry_record(
            &self.telemetry_context,
            &self.usage,
            finish_reason,
            error_type,
            self.saw_tool_delta,
        ));
    }
}

fn anthropic_stop_reason(reason: &ProviderFinishReason, saw_tool_delta: bool) -> &'static str {
    if saw_tool_delta && matches!(reason, ProviderFinishReason::Stop) {
        return "tool_use";
    }

    match reason {
        ProviderFinishReason::Stop => "end_turn",
        ProviderFinishReason::ToolCalls | ProviderFinishReason::FunctionCall => "tool_use",
        ProviderFinishReason::Length => "max_tokens",
        ProviderFinishReason::Other(_) => "end_turn",
    }
}

fn anthropic_tool_use_json(call: ToolCallRecord) -> serde_json::Value {
    serde_json::json!({
        "type": "tool_use",
        "id": call.tool_call_id.as_ref(),
        "name": call.tool_name,
        "input": call.arguments_json
    })
}

fn parse_openai_request(raw: serde_json::Value) -> Result<OpenAiChatRequest, ApiError> {
    serde_json::from_value(raw)
        .map_err(|error| ApiError::InvalidRequest(format!("invalid OpenAI request: {error}")))
}

fn parse_openai_completion_request(
    raw: serde_json::Value,
) -> Result<OpenAiCompletionRequest, ApiError> {
    serde_json::from_value(raw).map_err(|error| {
        ApiError::InvalidRequest(format!("invalid OpenAI completion request: {error}"))
    })
}

fn parse_anthropic_request(raw: serde_json::Value) -> Result<AnthropicMessagesRequest, ApiError> {
    serde_json::from_value(raw)
        .map_err(|error| ApiError::InvalidRequest(format!("invalid Anthropic request: {error}")))
}

fn openai_tool_response_format(request: &OpenAiChatRequest) -> OpenAiToolResponseFormat {
    if !request.functions.is_empty() || !request.function_call.is_null() {
        OpenAiToolResponseFormat::LegacyFunctions
    } else {
        OpenAiToolResponseFormat::Tools
    }
}

pub fn normalize_openai_chat(request: OpenAiChatRequest) -> Result<NormalizedRequest, ApiError> {
    normalize_openai_chat_with_context(request, &RequestContext::default())
}

fn normalize_openai_chat_with_context(
    request: OpenAiChatRequest,
    base_context: &RequestContext,
) -> Result<NormalizedRequest, ApiError> {
    if request.stream {
        return Err(ApiError::StreamUnsupported);
    }

    let request_context = base_context.with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let request_id = request_context.request_id();
    let messages_debug = serde_json::to_string(&request.messages_debug()).unwrap_or_default();
    let conversation_fingerprint =
        request_context.conversation_fingerprint("openai_chat", &tenant_id, &messages_debug);
    let scope = IsolationKey {
        tenant_id: tenant_id.clone(),
        request_id: request_id.clone(),
        conversation_fingerprint: conversation_fingerprint.clone(),
    };
    let provider_model = provider_model_for_request_model(&request.model);

    let mut media_artifacts = Vec::new();
    let mut normalized_messages = Vec::new();
    let mut tool_results = Vec::new();
    let thinking_enabled = openai_thinking_enabled(&request);
    let thinking_format = thinking_format(&provider_model, &request.metadata);
    let reasoning_effort = openai_reasoning_effort(&request)?;
    let tool_choice = openai_effective_tool_choice(&request)?;
    let parallel_tool_calls = request.parallel_tool_calls;
    let provider_options = openai_provider_options(&request);
    let mut last_legacy_function_call_id: Option<ToolCallId> = None;

    for message in request.messages {
        let role = normalize_role(&message.role);

        if message.role == "function" {
            let Some(name) = message.name.clone() else {
                return Err(ApiError::InvalidRequest(
                    "OpenAI function messages must include name".to_string(),
                ));
            };
            let tool_call_id = last_legacy_function_call_id
                .clone()
                .unwrap_or_else(|| legacy_function_call_id(&name));
            let result_json = content_to_tool_result_json(message.content.as_ref());
            tool_results.push(ToolResultRecord {
                tool_call_id: tool_call_id.clone(),
                scope: scope.clone(),
                result_sha256: sha256_hex(&result_json.to_string()),
                result_json,
                status: ToolResultStatus::Accepted,
            });
            normalized_messages.push(NormalizedMessage {
                role: MessageRole::Tool,
                content: vec![NormalizedContentPart::ToolResult { tool_call_id }],
            });
            continue;
        }

        if message.role == "tool" {
            let Some(tool_call_id) = message.tool_call_id.clone() else {
                return Err(ApiError::InvalidRequest(
                    "OpenAI tool messages must include tool_call_id".to_string(),
                ));
            };
            let result_json = content_to_tool_result_json(message.content.as_ref());
            tool_results.push(ToolResultRecord {
                tool_call_id: ToolCallId::from(tool_call_id.clone()),
                scope: scope.clone(),
                result_sha256: sha256_hex(&result_json.to_string()),
                result_json,
                status: ToolResultStatus::Accepted,
            });
            normalized_messages.push(NormalizedMessage {
                role,
                content: vec![NormalizedContentPart::ToolResult {
                    tool_call_id: ToolCallId::from(tool_call_id),
                }],
            });
            continue;
        }

        let mut content = Vec::new();
        match message.content {
            Some(OpenAiContent::Text(text)) => {
                content.push(NormalizedContentPart::Text { text });
            }
            Some(OpenAiContent::Parts(parts)) => {
                for part in parts {
                    match part {
                        OpenAiContentPart::Text { text } => {
                            content.push(NormalizedContentPart::Text { text });
                        }
                        OpenAiContentPart::ImageUrl { image_url } => {
                            let artifact =
                                media_from_openai_url(&scope, media_artifacts.len(), image_url.url);
                            let artifact_ref = ArtifactRef {
                                scope: scope.clone(),
                                artifact_id: artifact.id.clone(),
                            };
                            media_artifacts.push(artifact);
                            content.push(NormalizedContentPart::Image { artifact_ref });
                        }
                    }
                }
            }
            None => {}
        }

        if message.role == "assistant" {
            for call in message.tool_calls {
                content.push(NormalizedContentPart::ToolCall {
                    tool_call_id: ToolCallId::from(call.id),
                    tool_name: call.function.name,
                    arguments_json: openai_tool_arguments(call.function.arguments),
                });
            }
            if let Some(function_call) = message.function_call {
                let tool_call_id = legacy_function_call_id(&function_call.name);
                last_legacy_function_call_id = Some(tool_call_id.clone());
                content.push(NormalizedContentPart::ToolCall {
                    tool_call_id,
                    tool_name: function_call.name,
                    arguments_json: openai_tool_arguments(function_call.arguments),
                });
            }
        }

        normalized_messages.push(NormalizedMessage { role, content });
    }

    let tools = openai_tool_definitions(request.tools, request.functions);
    validate_tool_choice(&tool_choice, &tools)?;

    Ok(NormalizedRequest {
        request_id,
        tenant_id,
        conversation_fingerprint,
        source_format: SourceFormat::OpenAIChat,
        model: provider_model,
        messages: normalized_messages,
        media_artifacts,
        tools,
        tool_choice,
        parallel_tool_calls,
        tool_results,
        stream: false,
        thinking_enabled,
        thinking_format,
        reasoning_effort,
        public_reasoning_enabled: false,
        provider_options,
        metadata: request.metadata,
    })
}

pub fn normalize_anthropic_messages(
    request: AnthropicMessagesRequest,
) -> Result<NormalizedRequest, ApiError> {
    normalize_anthropic_messages_with_context(request, &RequestContext::default())
}

fn normalize_anthropic_messages_with_context(
    request: AnthropicMessagesRequest,
    base_context: &RequestContext,
) -> Result<NormalizedRequest, ApiError> {
    if request.stream {
        return Err(ApiError::StreamUnsupported);
    }

    let request_context = base_context.with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let request_id = request_context.request_id();
    let messages_debug = serde_json::to_string(&request.messages_debug()).unwrap_or_default();
    let conversation_fingerprint =
        request_context.conversation_fingerprint("anthropic_messages", &tenant_id, &messages_debug);
    let scope = IsolationKey {
        tenant_id: tenant_id.clone(),
        request_id: request_id.clone(),
        conversation_fingerprint: conversation_fingerprint.clone(),
    };
    let provider_model = provider_model_for_request_model(&request.model);

    let mut media_artifacts = Vec::new();
    let mut normalized_messages = Vec::new();
    let mut tool_results = Vec::new();
    let thinking_enabled = anthropic_thinking_enabled(&request);
    let thinking_format = thinking_format(&provider_model, &request.metadata);
    let reasoning_effort = anthropic_reasoning_effort(&request)?;
    let (tool_choice, parallel_tool_calls) = anthropic_tool_choice(&request.tool_choice)?;
    let provider_options = anthropic_provider_options(&request);

    if let Some(system) = request.system {
        normalized_messages.push(NormalizedMessage {
            role: MessageRole::System,
            content: anthropic_system_to_parts(system, &scope, &mut media_artifacts),
        });
    }

    for message in request.messages {
        let mut content = Vec::new();
        match message.content {
            AnthropicContent::Text(text) => {
                content.push(NormalizedContentPart::Text { text });
            }
            AnthropicContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        AnthropicContentBlock::Text { text } => {
                            content.push(NormalizedContentPart::Text { text });
                        }
                        AnthropicContentBlock::Image { source } => {
                            let artifact =
                                media_from_anthropic_source(&scope, media_artifacts.len(), source);
                            let artifact_ref = ArtifactRef {
                                scope: scope.clone(),
                                artifact_id: artifact.id.clone(),
                            };
                            media_artifacts.push(artifact);
                            content.push(NormalizedContentPart::Image { artifact_ref });
                        }
                        AnthropicContentBlock::ToolUse { id, name, input } => {
                            content.push(NormalizedContentPart::ToolCall {
                                tool_call_id: ToolCallId::from(id),
                                tool_name: name,
                                arguments_json: input,
                            });
                        }
                        AnthropicContentBlock::ToolResult {
                            tool_use_id,
                            content: result_json,
                        } => {
                            tool_results.push(ToolResultRecord {
                                tool_call_id: ToolCallId::from(tool_use_id.clone()),
                                scope: scope.clone(),
                                result_sha256: sha256_hex(&result_json.to_string()),
                                result_json,
                                status: ToolResultStatus::Accepted,
                            });
                            content.push(NormalizedContentPart::ToolResult {
                                tool_call_id: ToolCallId::from(tool_use_id),
                            });
                        }
                    }
                }
            }
        }

        normalized_messages.push(NormalizedMessage {
            role: normalize_role(&message.role),
            content,
        });
    }

    let tools = request
        .tools
        .into_iter()
        .map(|tool| ToolDefinition {
            name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema,
        })
        .collect::<Vec<_>>();
    validate_tool_choice(&tool_choice, &tools)?;

    Ok(NormalizedRequest {
        request_id,
        tenant_id,
        conversation_fingerprint,
        source_format: SourceFormat::AnthropicMessages,
        model: provider_model,
        messages: normalized_messages,
        media_artifacts,
        tools,
        tool_choice,
        parallel_tool_calls,
        tool_results,
        stream: false,
        thinking_enabled,
        thinking_format,
        reasoning_effort,
        public_reasoning_enabled: false,
        provider_options,
        metadata: request.metadata,
    })
}

fn openai_provider_options(request: &OpenAiChatRequest) -> serde_json::Value {
    let mut options = serde_json::Map::new();
    for (key, value) in &request.extra {
        if !is_gateway_extra_option(key) {
            options.insert(key.clone(), value.clone());
        }
    }
    if let Some(reasoning) = sanitized_reasoning_options(&request.reasoning) {
        options.insert("reasoning".to_string(), reasoning);
    }
    if let Some(metadata) = sanitized_provider_metadata(&request.metadata) {
        options.insert("metadata".to_string(), metadata);
    }
    serde_json::Value::Object(options)
}

fn anthropic_provider_options(request: &AnthropicMessagesRequest) -> serde_json::Value {
    let mut options = serde_json::Map::new();
    for (key, value) in &request.extra {
        if !is_gateway_extra_option(key) {
            options.insert(key.clone(), value.clone());
        }
    }
    if let Some(metadata) = sanitized_provider_metadata(&request.metadata) {
        options.insert("metadata".to_string(), metadata);
    }
    if !request.thinking.is_null() {
        options.insert("thinking".to_string(), request.thinking.clone());
    }
    serde_json::Value::Object(options)
}

fn is_gateway_extra_option(key: &str) -> bool {
    matches!(
        key,
        "include_reasoning"
            | "include_thinking"
            | "show_reasoning"
            | "return_reasoning"
            | "reasoning_content"
    )
}

fn sanitized_reasoning_options(value: &serde_json::Value) -> Option<serde_json::Value> {
    let object = value.as_object()?;
    let mut sanitized = object.clone();
    sanitized.remove("effort");
    if sanitized.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(sanitized))
    }
}

fn sanitized_provider_metadata(value: &serde_json::Value) -> Option<serde_json::Value> {
    let object = value.as_object()?;
    let mut sanitized = object.clone();
    for key in [
        "tenant_id",
        "tenant",
        "organization_id",
        "project_id",
        "request_id",
        "correlation_id",
        "conversation_id",
        "thread_id",
        "session_id",
        "context",
        "include_context_report",
        "include_encrypted_subagent_state",
        "include_reasoning",
        "include_thinking",
        "show_reasoning",
        "return_reasoning",
        "reasoning_content",
        "agent",
        "orchestration",
        "max_parallel_agents",
        "parallel_agents",
        "thinking_mode",
        "thinking_format",
    ] {
        sanitized.remove(key);
    }
    if sanitized.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(sanitized))
    }
}

fn openai_tool_choice(value: &serde_json::Value) -> Result<ToolChoice, ApiError> {
    if value.is_null() {
        return Ok(ToolChoice::Auto);
    }

    if let Some(choice) = value.as_str() {
        return match choice {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Ok(ToolChoice::Required),
            other => Err(ApiError::InvalidRequest(format!(
                "unsupported OpenAI tool_choice={other}"
            ))),
        };
    }

    let object = value.as_object().ok_or_else(|| {
        ApiError::InvalidRequest("OpenAI tool_choice must be a string or object".to_string())
    })?;
    if object.get("type").and_then(|value| value.as_str()) != Some("function") {
        return Err(ApiError::InvalidRequest(
            "OpenAI object tool_choice must use type=function".to_string(),
        ));
    }
    let name = object
        .get("function")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            ApiError::InvalidRequest(
                "OpenAI named tool_choice must include function.name".to_string(),
            )
        })?;
    Ok(ToolChoice::Named {
        name: name.to_string(),
    })
}

fn openai_effective_tool_choice(request: &OpenAiChatRequest) -> Result<ToolChoice, ApiError> {
    if !request.tool_choice.is_null() {
        return openai_tool_choice(&request.tool_choice);
    }
    openai_function_call_choice(&request.function_call)
}

fn openai_function_call_choice(value: &serde_json::Value) -> Result<ToolChoice, ApiError> {
    if value.is_null() {
        return Ok(ToolChoice::Auto);
    }

    if let Some(choice) = value.as_str() {
        return match choice {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            other => Err(ApiError::InvalidRequest(format!(
                "unsupported OpenAI function_call={other}"
            ))),
        };
    }

    let object = value.as_object().ok_or_else(|| {
        ApiError::InvalidRequest("OpenAI function_call must be a string or object".to_string())
    })?;
    let name = object
        .get("name")
        .and_then(|value| value.as_str())
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            ApiError::InvalidRequest("OpenAI function_call object must include name".to_string())
        })?;
    Ok(ToolChoice::Named {
        name: name.to_string(),
    })
}

fn openai_tool_definitions(
    tools: Vec<OpenAiTool>,
    functions: Vec<OpenAiFunctionTool>,
) -> Vec<ToolDefinition> {
    tools
        .into_iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name,
            description: tool.function.description,
            input_schema: tool.function.parameters,
        })
        .chain(functions.into_iter().map(|function| ToolDefinition {
            name: function.name,
            description: function.description,
            input_schema: function.parameters,
        }))
        .collect()
}

fn legacy_function_call_id(name: &str) -> ToolCallId {
    ToolCallId::from(format!("legacy-function-{name}"))
}

fn anthropic_tool_choice(
    value: &serde_json::Value,
) -> Result<(ToolChoice, Option<bool>), ApiError> {
    if value.is_null() {
        return Ok((ToolChoice::Auto, None));
    }

    if let Some(choice) = value.as_str() {
        return Ok((
            match choice {
                "auto" => ToolChoice::Auto,
                "none" => ToolChoice::None,
                "any" | "required" => ToolChoice::Required,
                other => {
                    return Err(ApiError::InvalidRequest(format!(
                        "unsupported Anthropic tool_choice={other}"
                    )));
                }
            },
            None,
        ));
    }

    let object = value.as_object().ok_or_else(|| {
        ApiError::InvalidRequest("Anthropic tool_choice must be a string or object".to_string())
    })?;
    let parallel_tool_calls = object
        .get("disable_parallel_tool_use")
        .and_then(|value| value.as_bool())
        .map(|disabled| !disabled);
    let choice_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            ApiError::InvalidRequest("Anthropic tool_choice must include type".to_string())
        })?;
    let choice = match choice_type {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "any" => ToolChoice::Required,
        "tool" => {
            let name = object
                .get("name")
                .and_then(|value| value.as_str())
                .filter(|name| !name.trim().is_empty())
                .ok_or_else(|| {
                    ApiError::InvalidRequest(
                        "Anthropic tool tool_choice must include name".to_string(),
                    )
                })?;
            ToolChoice::Named {
                name: name.to_string(),
            }
        }
        other => {
            return Err(ApiError::InvalidRequest(format!(
                "unsupported Anthropic tool_choice.type={other}"
            )));
        }
    };
    Ok((choice, parallel_tool_calls))
}

fn validate_tool_choice(choice: &ToolChoice, tools: &[ToolDefinition]) -> Result<(), ApiError> {
    match choice {
        ToolChoice::Auto | ToolChoice::None => Ok(()),
        ToolChoice::Required if tools.is_empty() => Err(ApiError::InvalidRequest(
            "tool_choice=required requires at least one tool".to_string(),
        )),
        ToolChoice::Required => Ok(()),
        ToolChoice::Named { name } if tools.iter().any(|tool| tool.name == *name) => Ok(()),
        ToolChoice::Named { name } => Err(ApiError::InvalidRequest(format!(
            "unknown tool_choice named tool: {name}"
        ))),
    }
}

fn openai_thinking_enabled(request: &OpenAiChatRequest) -> bool {
    if openai_thinking_explicitly_disabled(request) {
        return false;
    }

    request.enable_thinking == Some(true)
        || request.preserve_thinking == Some(true)
        || request
            .metadata
            .get("thinking_mode")
            .and_then(|value| value.as_bool())
            == Some(true)
        || request
            .chat_template_kwargs
            .get("enable_thinking")
            .and_then(|value| value.as_bool())
            == Some(true)
        || request
            .thinking
            .get("type")
            .and_then(|value| value.as_str())
            == Some("enabled")
        || request
            .thinking
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(true)
        || request
            .reasoning
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(true)
        || is_gemma_system_token_model(&request.model)
}

fn openai_thinking_explicitly_disabled(request: &OpenAiChatRequest) -> bool {
    request.enable_thinking == Some(false)
        || request.preserve_thinking == Some(false)
        || request
            .metadata
            .get("thinking_mode")
            .and_then(|value| value.as_bool())
            == Some(false)
        || request
            .chat_template_kwargs
            .get("enable_thinking")
            .and_then(|value| value.as_bool())
            == Some(false)
        || request
            .thinking
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(false)
        || request
            .reasoning
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(false)
}

fn openai_reasoning_effort(request: &OpenAiChatRequest) -> Result<ReasoningEffort, ApiError> {
    if let Some((_, effort)) = split_effort_model_alias(&request.model) {
        return Ok(effort);
    }

    reasoning_effort_from_values(&[
        &request.reasoning,
        request
            .metadata
            .get("reasoning")
            .unwrap_or(&serde_json::Value::Null),
        &request.metadata,
    ])
}

fn anthropic_reasoning_effort(
    request: &AnthropicMessagesRequest,
) -> Result<ReasoningEffort, ApiError> {
    if let Some((_, effort)) = split_effort_model_alias(&request.model) {
        return Ok(effort);
    }

    reasoning_effort_from_values(&[
        &request.reasoning,
        request
            .metadata
            .get("reasoning")
            .unwrap_or(&serde_json::Value::Null),
        &request.thinking,
        &request.metadata,
    ])
}

fn reasoning_effort_from_values(
    values: &[&serde_json::Value],
) -> Result<ReasoningEffort, ApiError> {
    for value in values {
        if let Some(effort) = value
            .get("effort")
            .or_else(|| value.get("reasoning_effort"))
            .and_then(|effort| effort.as_str())
        {
            return parse_reasoning_effort(effort);
        }
    }
    Ok(ReasoningEffort::Medium)
}

fn parse_reasoning_effort(value: &str) -> Result<ReasoningEffort, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(ReasoningEffort::None),
        "low" => Ok(ReasoningEffort::Low),
        "medium" => Ok(ReasoningEffort::Medium),
        "high" => Ok(ReasoningEffort::High),
        "xhigh" | "x_high" | "extra_high" => Ok(ReasoningEffort::XHigh),
        other => Err(ApiError::InvalidRequest(format!(
            "unsupported reasoning.effort={other}; expected none, low, medium, high, or xhigh"
        ))),
    }
}

fn provider_model_for_request_model(model: &str) -> String {
    split_effort_model_alias(model)
        .map(|(base_model, _)| base_model)
        .unwrap_or_else(|| model.to_string())
}

fn split_effort_model_alias(model: &str) -> Option<(String, ReasoningEffort)> {
    let trimmed = model.trim();
    for (suffix, effort) in effort_model_suffixes() {
        let Some(base_model) = trimmed.strip_suffix(suffix) else {
            continue;
        };
        if base_model.is_empty() || !model_supports_effort_aliases(base_model) {
            continue;
        }
        return Some((base_model.to_string(), effort.clone()));
    }
    None
}

fn effort_model_suffixes() -> &'static [(&'static str, ReasoningEffort)] {
    &[
        ("-none", ReasoningEffort::None),
        ("-low", ReasoningEffort::Low),
        ("-medium", ReasoningEffort::Medium),
        ("-high", ReasoningEffort::High),
        ("-xhigh", ReasoningEffort::XHigh),
    ]
}

fn anthropic_thinking_enabled(request: &AnthropicMessagesRequest) -> bool {
    if request
        .metadata
        .get("thinking_mode")
        .and_then(|value| value.as_bool())
        == Some(false)
        || request
            .thinking
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(false)
    {
        return false;
    }

    request
        .metadata
        .get("thinking_mode")
        .and_then(|value| value.as_bool())
        == Some(true)
        || request
            .thinking
            .get("type")
            .and_then(|value| value.as_str())
            == Some("enabled")
        || request
            .thinking
            .get("enabled")
            .and_then(|value| value.as_bool())
            == Some(true)
        || is_gemma_system_token_model(&request.model)
}

fn thinking_format(model: &str, metadata: &serde_json::Value) -> ThinkingFormat {
    match metadata
        .get("thinking_format")
        .and_then(|value| value.as_str())
    {
        Some("qwen_dashscope") => ThinkingFormat::QwenDashScope,
        Some("qwen_chat_template") => ThinkingFormat::QwenChatTemplate,
        Some("gemma_system_token") => ThinkingFormat::GemmaSystemToken,
        _ if is_gemma_system_token_model(model) => ThinkingFormat::GemmaSystemToken,
        _ if model.to_lowercase().contains("gemma") => ThinkingFormat::GemmaSystemToken,
        _ if model.to_lowercase().contains("qwen") => ThinkingFormat::QwenChatTemplate,
        _ => ThinkingFormat::Auto,
    }
}

fn is_gemma_system_token_model(model: &str) -> bool {
    let provider_model = provider_model_for_request_model(model);
    provider_model.to_lowercase().contains("gemma")
        || model_list_from_env(&[
            "MIYA_GEMMA_MODELS",
            "MULTI_AGENT_GEMMA_MODELS",
            "GEMMA_MODELS",
        ])
        .iter()
        .any(|configured| configured.eq_ignore_ascii_case(&provider_model))
}

impl OpenAiChatRequest {
    fn messages_debug(&self) -> String {
        format!("{:?}", self.messages)
    }
}

impl OpenAiCompletionRequest {
    fn into_chat_request(self) -> Result<OpenAiChatRequest, ApiError> {
        let prompt = completion_prompt_text(&self.prompt)?;
        Ok(OpenAiChatRequest {
            model: self.model,
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(OpenAiContent::Text(prompt)),
                tool_calls: Vec::new(),
                function_call: None,
                tool_call_id: None,
                name: None,
            }],
            tools: Vec::new(),
            tool_choice: serde_json::Value::Null,
            functions: Vec::new(),
            function_call: serde_json::Value::Null,
            parallel_tool_calls: None,
            thinking: self.thinking,
            reasoning: self.reasoning,
            chat_template_kwargs: self.chat_template_kwargs,
            enable_thinking: self.enable_thinking,
            preserve_thinking: self.preserve_thinking,
            stream: self.stream,
            metadata: self.metadata,
            extra: self.extra,
        })
    }
}

fn completion_prompt_text(value: &serde_json::Value) -> Result<String, ApiError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }

    if let Some(parts) = value.as_array()
        && parts.iter().all(serde_json::Value::is_string)
    {
        return Ok(parts
            .iter()
            .filter_map(|part| part.as_str())
            .collect::<Vec<_>>()
            .join("\n"));
    }

    Err(ApiError::InvalidRequest(
        "OpenAI completions prompt must be a string or string array".to_string(),
    ))
}

impl AnthropicMessagesRequest {
    fn messages_debug(&self) -> String {
        format!("{:?}", self.messages)
    }
}

fn normalize_role(role: &str) -> MessageRole {
    match role {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

fn content_to_tool_result_json(content: Option<&OpenAiContent>) -> serde_json::Value {
    match content {
        Some(OpenAiContent::Text(text)) => {
            serde_json::from_str(text).unwrap_or_else(|_| serde_json::Value::String(text.clone()))
        }
        Some(OpenAiContent::Parts(_)) | None => serde_json::Value::Null,
    }
}

fn anthropic_system_to_parts(
    system: AnthropicSystem,
    scope: &IsolationKey,
    media_artifacts: &mut Vec<MediaArtifact>,
) -> Vec<NormalizedContentPart> {
    match system {
        AnthropicSystem::Text(text) => vec![NormalizedContentPart::Text { text }],
        AnthropicSystem::Blocks(blocks) => blocks
            .into_iter()
            .filter_map(|block| match block {
                AnthropicContentBlock::Text { text } => Some(NormalizedContentPart::Text { text }),
                AnthropicContentBlock::Image { source } => {
                    let artifact =
                        media_from_anthropic_source(scope, media_artifacts.len(), source);
                    let artifact_ref = ArtifactRef {
                        scope: scope.clone(),
                        artifact_id: artifact.id.clone(),
                    };
                    media_artifacts.push(artifact);
                    Some(NormalizedContentPart::Image { artifact_ref })
                }
                AnthropicContentBlock::ToolUse { .. } => None,
                AnthropicContentBlock::ToolResult { .. } => None,
            })
            .collect(),
    }
}

fn openai_tool_arguments(arguments: serde_json::Value) -> serde_json::Value {
    match arguments {
        serde_json::Value::String(text) => {
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))
        }
        other => other,
    }
}

fn media_from_openai_url(scope: &IsolationKey, index: usize, url: String) -> MediaArtifact {
    let (media_type, source) = if url.starts_with("data:") {
        (
            media_type_from_data_url(&url),
            MediaSource::DataUrl {
                data_url: url.clone(),
            },
        )
    } else {
        (
            "image/unknown".to_string(),
            MediaSource::RemoteUrl { url: url.clone() },
        )
    };

    MediaArtifact {
        id: ArtifactId::from(format!("media-{index}")),
        scope: scope.clone(),
        media_type,
        source,
        sha256: sha256_hex(&url),
        byte_len: Some(url.len() as u64),
    }
}

fn media_from_anthropic_source(
    scope: &IsolationKey,
    index: usize,
    source: AnthropicImageSource,
) -> MediaArtifact {
    match source {
        AnthropicImageSource::Base64 { media_type, data } => MediaArtifact {
            id: ArtifactId::from(format!("media-{index}")),
            scope: scope.clone(),
            media_type,
            sha256: sha256_hex(&data),
            byte_len: Some(data.len() as u64),
            source: MediaSource::Base64 { data },
        },
        AnthropicImageSource::Url { url } => MediaArtifact {
            id: ArtifactId::from(format!("media-{index}")),
            scope: scope.clone(),
            media_type: "image/unknown".to_string(),
            sha256: sha256_hex(&url),
            byte_len: Some(url.len() as u64),
            source: MediaSource::RemoteUrl { url },
        },
    }
}

fn media_type_from_data_url(url: &str) -> String {
    url.strip_prefix("data:")
        .and_then(|rest| rest.split_once(';').map(|(media_type, _)| media_type))
        .filter(|media_type| !media_type.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string()
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn crate_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{MediaSource, NormalizedContentPart, SourceFormat};

    #[test]
    fn normalization_openai_text_and_image_url() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]
            }]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.source_format, SourceFormat::OpenAIChat);
        assert_eq!(normalized.media_artifacts.len(), 1);
        assert!(matches!(
            normalized.media_artifacts[0].source,
            MediaSource::DataUrl { .. }
        ));
        assert!(matches!(
            normalized.messages[0].content[1],
            NormalizedContentPart::Image { .. }
        ));
    }

    #[test]
    fn normalization_openai_tool_result() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "{\"ok\":true}"
            }]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.tool_results.len(), 1);
        assert_eq!(normalized.tool_results[0].tool_call_id.as_ref(), "call-1");
    }

    #[test]
    fn normalization_openai_named_tool_choice() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "parallel_tool_calls": false,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"}
                }
            }],
            "messages": [{"role": "user", "content": "use lookup"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(
            normalized.tool_choice,
            ToolChoice::Named {
                name: "lookup".to_string()
            }
        );
        assert_eq!(normalized.parallel_tool_calls, Some(false));
    }

    #[test]
    fn normalization_openai_legacy_functions_map_to_tools_and_function_call() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "function_call": {"name": "lookup"},
            "functions": [{
                "name": "lookup",
                "description": "legacy lookup",
                "parameters": {"type": "object"}
            }],
            "messages": [{"role": "user", "content": "use lookup"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.tools.len(), 1);
        assert_eq!(normalized.tools[0].name, "lookup");
        assert_eq!(
            normalized.tool_choice,
            ToolChoice::Named {
                name: "lookup".to_string()
            }
        );
    }

    #[test]
    fn normalization_openai_preserves_assistant_tool_calls() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"city\":\"Taipei\"}"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "{\"temperature\":\"24C\"}"}
            ]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert!(matches!(
            &normalized.messages[1].content[0],
            NormalizedContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments_json
            } if tool_call_id.as_ref() == "call-1"
                && tool_name == "lookup"
                && arguments_json["city"] == "Taipei"
        ));
        assert_eq!(normalized.tool_results.len(), 1);
    }

    #[test]
    fn normalization_openai_preserves_legacy_function_call_history() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": null,
                    "function_call": {
                        "name": "lookup",
                        "arguments": "{\"city\":\"Taipei\"}"
                    }
                },
                {"role": "function", "name": "lookup", "content": "{\"temperature\":\"24C\"}"}
            ]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert!(matches!(
            &normalized.messages[1].content[0],
            NormalizedContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments_json
            } if tool_call_id.as_ref() == "legacy-function-lookup"
                && tool_name == "lookup"
                && arguments_json["city"] == "Taipei"
        ));
        assert_eq!(
            normalized.tool_results[0].tool_call_id.as_ref(),
            "legacy-function-lookup"
        );
    }

    #[test]
    fn normalization_openai_rejects_unknown_named_tool_choice() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "tool_choice": {"type": "function", "function": {"name": "missing"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"}
                }
            }],
            "messages": [{"role": "user", "content": "use lookup"}]
        }))
        .unwrap();

        assert!(matches!(
            normalize_openai_chat(request),
            Err(ApiError::InvalidRequest(message)) if message.contains("unknown tool_choice")
        ));
    }

    #[test]
    fn normalization_anthropic_text_and_base64_image() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]
            }]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.source_format, SourceFormat::AnthropicMessages);
        assert_eq!(normalized.media_artifacts.len(), 1);
        assert!(matches!(
            normalized.media_artifacts[0].source,
            MediaSource::Base64 { .. }
        ));
    }

    #[test]
    fn normalization_anthropic_text_and_url_image() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
                    {"type": "text", "text": "inspect"}
                ]
            }]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.media_artifacts.len(), 1);
        assert!(matches!(
            normalized.media_artifacts[0].source,
            MediaSource::RemoteUrl { .. }
        ));
    }

    #[test]
    fn normalization_anthropic_system_text_and_blocks_are_preserved() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "system": [
                {"type": "text", "text": "Answer in Traditional Chinese."}
            ],
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.messages[0].role, MessageRole::System);
        assert!(matches!(
            &normalized.messages[0].content[0],
            NormalizedContentPart::Text { text } if text == "Answer in Traditional Chinese."
        ));
    }

    #[test]
    fn normalization_anthropic_tool_result() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": {"ok": true}}
                ]
            }]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.tool_results.len(), 1);
        assert_eq!(normalized.tool_results[0].tool_call_id.as_ref(), "call-1");
    }

    #[test]
    fn normalization_anthropic_tool_choice_any_and_parallel_disable() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "tools": [{
                "name": "lookup",
                "input_schema": {"type": "object"}
            }],
            "messages": [{"role": "user", "content": "use lookup"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.tool_choice, ToolChoice::Required);
        assert_eq!(normalized.parallel_tool_calls, Some(false));
    }

    #[test]
    fn normalization_anthropic_tool_choice_named_none_and_validation() {
        let named: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "tool_choice": {"type": "tool", "name": "lookup"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "use lookup"}]
        }))
        .unwrap();
        assert_eq!(
            normalize_anthropic_messages(named).unwrap().tool_choice,
            ToolChoice::Named {
                name: "lookup".to_string()
            }
        );

        let none: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "tool_choice": {"type": "none"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "do not use tools"}]
        }))
        .unwrap();
        assert_eq!(
            normalize_anthropic_messages(none).unwrap().tool_choice,
            ToolChoice::None
        );

        let missing: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "tool_choice": {"type": "tool", "name": "missing"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "use missing"}]
        }))
        .unwrap();
        assert!(matches!(
            normalize_anthropic_messages(missing),
            Err(ApiError::InvalidRequest(message)) if message.contains("unknown tool_choice")
        ));

        let required_without_tools: AnthropicMessagesRequest =
            serde_json::from_value(serde_json::json!({
                "model": "mock",
                "max_tokens": 256,
                "tool_choice": {"type": "any"},
                "messages": [{"role": "user", "content": "use a tool"}]
            }))
            .unwrap();
        assert!(matches!(
            normalize_anthropic_messages(required_without_tools),
            Err(ApiError::InvalidRequest(message)) if message.contains("requires at least one tool")
        ));
    }

    #[test]
    fn normalization_anthropic_preserves_assistant_tool_use() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {"city": "Taipei"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": {"temperature": "24C"}
                    }]
                }
            ]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert!(matches!(
            &normalized.messages[1].content[0],
            NormalizedContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments_json
            } if tool_call_id.as_ref() == "call-1"
                && tool_name == "lookup"
                && arguments_json["city"] == "Taipei"
        ));
        assert_eq!(normalized.tool_results.len(), 1);
    }

    #[test]
    fn normalization_stream_true_is_rejected() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        assert!(matches!(
            normalize_openai_chat(request),
            Err(ApiError::StreamUnsupported)
        ));
    }

    #[test]
    fn normalization_openai_enables_thinking_mode_from_metadata() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "metadata": {"thinking_mode": true},
            "messages": [{"role": "user", "content": "think carefully"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert!(normalized.thinking_enabled);
    }

    #[test]
    fn normalization_openai_maps_gemma_named_models_to_gemma_generation_format() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "local-gemma-finetune",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.thinking_format, ThinkingFormat::GemmaSystemToken);
        assert!(normalized.thinking_enabled);
    }

    #[test]
    fn normalization_anthropic_enables_thinking_mode_from_thinking_config() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "thinking": {"type": "enabled", "budget_tokens": 128},
            "messages": [{"role": "user", "content": "think carefully"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert!(normalized.thinking_enabled);
    }

    #[test]
    fn normalization_openai_maps_reasoning_effort_to_orchestration_policy() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "reasoning": {"effort": "high"},
            "messages": [{"role": "user", "content": "use more agents"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::High);
        assert_eq!(normalized.reasoning_effort.max_agents(), 32);
    }

    #[test]
    fn normalization_openai_defaults_unconfigured_reasoning_effort_to_medium_tier() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "use default effort"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::Medium);
        assert_eq!(normalized.reasoning_effort.max_agents(), 16);
    }

    #[test]
    fn normalization_openai_model_alias_overrides_reasoning_effort_and_uses_provider_model() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "local-gemma-finetune-high",
            "reasoning": {"effort": "low"},
            "messages": [{"role": "user", "content": "use the model tier"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.model, "local-gemma-finetune");
        assert_eq!(normalized.reasoning_effort, ReasoningEffort::High);
        assert_eq!(normalized.reasoning_effort.max_agents(), 32);
        assert_eq!(normalized.thinking_format, ThinkingFormat::GemmaSystemToken);
    }

    #[test]
    fn normalization_openai_model_aliases_are_not_tied_to_a_hardcoded_model_id() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "custom-model-high",
            "messages": [{"role": "user", "content": "use the model tier"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.model, "custom-model");
        assert_eq!(normalized.reasoning_effort, ReasoningEffort::High);
    }

    #[test]
    fn normalization_anthropic_model_alias_overrides_reasoning_effort_and_uses_provider_model() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "local-gemma-finetune-xhigh",
            "thinking": {"type": "enabled", "effort": "low"},
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "use the model tier"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.model, "local-gemma-finetune");
        assert_eq!(normalized.reasoning_effort, ReasoningEffort::XHigh);
        assert_eq!(normalized.reasoning_effort.max_agents(), 64);
        assert_eq!(normalized.thinking_format, ThinkingFormat::GemmaSystemToken);
    }

    #[test]
    fn normalization_openai_preserves_model_config_provider_options() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "temperature": 0.7,
            "top_p": 0.4,
            "max_completion_tokens": 321,
            "response_format": {"type": "json_object"},
            "seed": 42,
            "stream_options": {"include_usage": true},
            "reasoning_effort": "low",
            "reasoning": {"effort": "high", "summary": "auto"},
            "metadata": {
                "tenant_id": "tenant-a",
                "context": {"id": "ctx-a"},
                "foo": "bar"
            },
            "messages": [{"role": "user", "content": "config passthrough"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::High);
        assert_eq!(normalized.provider_options["temperature"], 0.7);
        assert_eq!(normalized.provider_options["top_p"], 0.4);
        assert_eq!(normalized.provider_options["max_completion_tokens"], 321);
        assert_eq!(
            normalized.provider_options["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        assert_eq!(normalized.provider_options["seed"], 42);
        assert_eq!(
            normalized.provider_options["stream_options"],
            serde_json::json!({"include_usage": true})
        );
        assert_eq!(normalized.provider_options["reasoning_effort"], "low");
        assert_eq!(normalized.provider_options["reasoning"]["summary"], "auto");
        assert!(
            normalized.provider_options["reasoning"]
                .get("effort")
                .is_none()
        );
        assert_eq!(normalized.provider_options["metadata"]["foo"], "bar");
        assert!(
            normalized.provider_options["metadata"]
                .get("tenant_id")
                .is_none()
        );
        assert!(
            normalized.provider_options["metadata"]
                .get("context")
                .is_none()
        );
    }

    #[test]
    fn normalization_openai_treats_include_reasoning_as_gateway_output_flag() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "include_reasoning": true,
            "metadata": {
                "include_reasoning": true,
                "foo": "bar"
            },
            "messages": [{"role": "user", "content": "show public reasoning"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert!(
            normalized
                .provider_options
                .get("include_reasoning")
                .is_none()
        );
        assert_eq!(normalized.provider_options["metadata"]["foo"], "bar");
        assert!(
            normalized.provider_options["metadata"]
                .get("include_reasoning")
                .is_none()
        );
    }

    #[test]
    fn public_reasoning_mode_parses_deployment_policy_values() {
        assert_eq!(
            PublicReasoningMode::from_env_value(Some("always")),
            PublicReasoningMode::Always
        );
        assert_eq!(
            PublicReasoningMode::from_env_value(Some("strip")),
            PublicReasoningMode::Never
        );
        assert_eq!(
            PublicReasoningMode::from_env_value(Some("request")),
            PublicReasoningMode::Request
        );
        assert_eq!(
            PublicReasoningMode::from_env_value(None),
            PublicReasoningMode::Always
        );
        assert!(PublicReasoningMode::Always.resolve(false));
        assert!(!PublicReasoningMode::Never.resolve(true));
        assert!(PublicReasoningMode::Request.resolve(true));
        assert!(!PublicReasoningMode::Request.resolve(false));
    }

    #[test]
    fn text_with_public_reasoning_preserves_structured_final_text_formatting() {
        let final_text =
            "<doc>\n<section>\n### Heading\n<details><hr>\nValue: 1\n</details>\n</section>";
        let wrapped = text_with_public_reasoning(
            final_text,
            "Multi-agent process summary:\n- Worker agent checked formatting.",
            &ThinkingFormat::QwenChatTemplate,
        );

        assert!(wrapped.contains("\n</think>\n\n<doc>\n<section>\n"));
        assert!(wrapped.contains("\n### Heading\n<details><hr>\n"));
        assert!(wrapped.contains("\nValue: 1\n</details>\n</section>"));
        assert!(!wrapped.contains("<doc><section>###Heading"));
    }

    #[test]
    fn normalization_openai_uses_metadata_identity_for_multi_tenant_scope() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "metadata": {
                "tenant_id": "tenant-a",
                "request_id": "request-a",
                "conversation_id": "conversation-a"
            },
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let base_context = RequestContext {
            tenant_id: Some("tenant-from-header".to_string()),
            request_id: Some("request-from-header".to_string()),
            conversation_id: Some("conversation-from-header".to_string()),
        };

        let normalized = normalize_openai_chat_with_context(request, &base_context).unwrap();
        let same_conversation: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "metadata": {
                "tenant_id": "tenant-a",
                "conversation_id": "conversation-a"
            },
            "messages": [{"role": "user", "content": "different text"}]
        }))
        .unwrap();
        let same_conversation =
            normalize_openai_chat_with_context(same_conversation, &RequestContext::default())
                .unwrap();

        assert_eq!(normalized.tenant_id.as_ref(), "tenant-a");
        assert_eq!(normalized.request_id.as_ref(), "request-a");
        assert_eq!(
            normalized.conversation_fingerprint,
            same_conversation.conversation_fingerprint
        );
    }

    #[test]
    fn normalization_anthropic_uses_header_identity_when_metadata_absent() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let base_context = RequestContext {
            tenant_id: Some("tenant-from-header".to_string()),
            request_id: Some("request-from-header".to_string()),
            conversation_id: Some("conversation-from-header".to_string()),
        };

        let normalized = normalize_anthropic_messages_with_context(request, &base_context).unwrap();

        assert_eq!(normalized.tenant_id.as_ref(), "tenant-from-header");
        assert_eq!(normalized.request_id.as_ref(), "request-from-header");
    }

    #[test]
    fn tenant_concurrency_limiter_defaults_to_medium_tier_when_unconfigured() {
        let limiter = TenantConcurrencyLimiter::from_env_value(None);

        assert_eq!(
            limiter.max_per_tenant,
            Some(DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS)
        );
        assert_eq!(
            DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS,
            ReasoningEffort::Medium.max_agents() as usize
        );
    }

    #[test]
    fn normalization_anthropic_accepts_reasoning_effort_none() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": "direct"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::None);
        assert!(normalized.reasoning_effort.is_direct());
    }

    #[test]
    fn direct_openai_sanitizer_disables_default_qwen_thinking() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "local-qwen-model",
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": "hello"}]
        }));

        assert!(sanitized.get("reasoning").is_none());
        assert_eq!(
            sanitized["chat_template_kwargs"]["enable_thinking"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn direct_openai_sanitizer_preserves_gemma_thinking_prompt() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "local-gemma-finetune",
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": "hello"}]
        }));

        assert!(sanitized.get("reasoning").is_none());
        assert!(sanitized.get("chat_template_kwargs").is_none());
        assert_eq!(sanitized["messages"][0]["role"], "system");
        assert!(
            sanitized["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("<|think|>")
        );
    }

    #[test]
    fn direct_openai_sanitizer_preserves_model_config_and_sanitizes_gateway_metadata() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "mock",
            "temperature": 0.75,
            "top_p": 0.5,
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "none", "summary": "auto"},
            "metadata": {
                "tenant_id": "tenant-a",
                "context": {"id": "ctx-a"},
                "foo": "bar"
            },
            "messages": [{"role": "user", "content": "direct config"}]
        }));

        assert_eq!(sanitized["temperature"], 0.75);
        assert_eq!(sanitized["top_p"], 0.5);
        assert_eq!(
            sanitized["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        assert_eq!(sanitized["reasoning"]["summary"], "auto");
        assert!(sanitized["reasoning"].get("effort").is_none());
        assert_eq!(sanitized["metadata"]["foo"], "bar");
        assert!(sanitized["metadata"].get("tenant_id").is_none());
        assert!(sanitized["metadata"].get("context").is_none());
    }

    #[test]
    fn direct_openai_sanitizer_adapts_gemma_named_tool_choice() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "local-gemma-finetune",
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {"type": "object"}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_news",
                        "parameters": {"type": "object"}
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_weather"}
            }
        }));

        assert_eq!(sanitized["tool_choice"], "required");
        assert_eq!(sanitized["tools"].as_array().unwrap().len(), 1);
        assert_eq!(sanitized["tools"][0]["function"]["name"], "lookup_weather");
        assert_eq!(
            sanitized["tools"][0]["function"]["parameters"]["properties"],
            serde_json::json!({})
        );
    }

    #[test]
    fn direct_openai_response_strips_reasoning_field() {
        let response = strip_direct_openai_response(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "final",
                    "reasoning": "private"
                }
            }]
        }));

        assert!(response["choices"][0]["message"].get("reasoning").is_none());
        assert_eq!(response["choices"][0]["message"]["content"], "final");
    }

    #[test]
    fn direct_openai_response_strips_thinking_channel_from_content() {
        let response = strip_direct_openai_response(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<|channel>thought\nprivate<channel|>public answer",
                    "reasoning_content": "private"
                }
            }]
        }));

        assert!(
            response["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
        assert_eq!(
            response["choices"][0]["message"]["content"],
            "public answer"
        );
    }

    #[test]
    fn direct_openai_response_strips_gemma_generation_wrappers() {
        let response = strip_direct_openai_response(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<bos><start_of_turn>model\n可直接給使用者的答案。<end_of_turn><eos>"
                }
            }]
        }));

        assert_eq!(
            response["choices"][0]["message"]["content"],
            "可直接給使用者的答案。"
        );
    }

    #[tokio::test]
    async fn routes_models_exposes_mock_by_default() {
        let response = tower::ServiceExt::oneshot(
            build_router(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let ids = value["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|model| model["id"].as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"mock"));
    }

    #[test]
    fn configured_model_ids_include_effort_aliases_without_duplicates() {
        let ids = configured_model_ids_from(Some("local-qwen-model, custom-model"));

        assert!(ids.contains(&"local-qwen-model".to_string()));
        assert!(ids.contains(&"local-qwen-model-high".to_string()));
        assert!(ids.contains(&"local-qwen-model-xhigh".to_string()));
        assert!(ids.contains(&"custom-model".to_string()));
        assert!(ids.contains(&"custom-model-none".to_string()));
        assert!(ids.contains(&"custom-model-low".to_string()));
        assert!(ids.contains(&"custom-model-medium".to_string()));
        assert!(ids.contains(&"custom-model-high".to_string()));
        assert!(ids.contains(&"custom-model-xhigh".to_string()));
        assert!(ids.contains(&"mock".to_string()));
        assert!(!ids.contains(&"mock-high".to_string()));
        assert_eq!(
            ids.iter()
                .filter(|id| id.as_str() == "custom-model")
                .count(),
            1
        );
    }

    #[test]
    fn raw_request_model_alias_is_rewritten_for_direct_provider_calls() {
        let raw = serde_json::json!({
            "model": "custom-model-none",
            "messages": [{"role": "user", "content": "direct"}]
        });

        let rewritten = raw_request_with_provider_model(raw, "custom-model-none");

        assert_eq!(rewritten["model"], "custom-model");
    }

    #[tokio::test]
    async fn routes_openai_chat_completion_returns_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert!(
            value["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("clear, usable answer")
        );
    }

    #[tokio::test]
    async fn routes_openai_response_reports_provider_usage() {
        let app = build_router_with_provider(std::sync::Arc::new(UsageRouteProvider));
        let value = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "messages": [{"role": "user", "content": "usage please"}]
            }),
        )
        .await;

        assert_eq!(value["usage"]["prompt_tokens"], 289);
        assert_eq!(value["usage"]["completion_tokens"], 391);
        assert_eq!(value["usage"]["total_tokens"], 680);
    }

    #[tokio::test]
    async fn routes_openai_chat_completion_returns_tool_call_shape() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"}
                            }
                        }],
                        "messages": [{"role": "user", "content": "please use a tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
    }

    #[tokio::test]
    async fn routes_openai_legacy_functions_return_function_call_shape() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "functions": [{
                            "name": "lookup",
                            "parameters": {"type": "object"}
                        }],
                        "function_call": "auto",
                        "messages": [{"role": "user", "content": "please use a tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["choices"][0]["finish_reason"], "function_call");
        assert_eq!(
            value["choices"][0]["message"]["function_call"]["name"],
            "lookup"
        );
        assert!(value["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn routes_openai_legacy_function_result_returns_final_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "functions": [{
                            "name": "lookup",
                            "parameters": {"type": "object"}
                        }],
                        "messages": [
                            {"role": "user", "content": "please use a tool"},
                            {"role": "assistant", "content": null, "function_call": {"name": "lookup", "arguments": "{\"query\":\"required\"}"}},
                            {"role": "function", "name": "lookup", "content": "{\"answer\":\"42\"}"}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert!(
            value["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("42")
        );
    }

    #[tokio::test]
    async fn routes_openai_tool_result_returns_final_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"}
                            }
                        }],
                        "messages": [
                            {"role": "user", "content": "please use a tool"},
                            {"role": "tool", "tool_call_id": "call-1", "content": "{\"answer\":\"42\"}"}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        let content = value["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("42"));
        assert!(value["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn routes_anthropic_tool_result_returns_final_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "tools": [{
                            "name": "lookup",
                            "input_schema": {"type": "object"}
                        }],
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "please use a tool"},
                                {"type": "tool_result", "tool_use_id": "call-1", "content": {"answer": "42"}}
                            ]
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["stop_reason"], "end_turn");
        assert!(value["content"][0]["text"].as_str().unwrap().contains("42"));
        assert_ne!(value["content"][0]["type"], "tool_use");
    }

    #[tokio::test]
    async fn routes_openai_reasoning_effort_none_bypasses_kernel() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "reasoning": {"effort": "none"},
                        "messages": [{"role": "user", "content": "direct path"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert!(
            value["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("direct backend response: direct path")
        );
        assert_eq!(provider.max_in_flight(), 0);
    }

    #[tokio::test]
    async fn routes_anthropic_reasoning_effort_none_bypasses_kernel() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "reasoning": {"effort": "none"},
                        "messages": [{"role": "user", "content": "direct anthropic path"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert!(
            value["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("direct backend response: direct anthropic path")
        );
        assert_eq!(provider.max_in_flight(), 0);
    }

    #[tokio::test]
    async fn routes_anthropic_messages_returns_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["type"], "message");
        assert_eq!(value["content"][0]["type"], "text");
    }

    #[tokio::test]
    async fn routes_anthropic_messages_includes_public_thinking_summary_when_requested() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "include_reasoning": true,
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["content"][0]["type"], "thinking");
        assert!(
            value["content"][0]["thinking"]
                .as_str()
                .unwrap()
                .contains("Multi-agent process summary")
        );
        assert_eq!(value["content"][1]["type"], "text");
        assert!(
            value["content"][1]["text"]
                .as_str()
                .unwrap()
                .contains("based on the verified agent results")
        );
    }

    #[tokio::test]
    async fn routes_anthropic_response_reports_provider_usage() {
        let app = build_router_with_provider(std::sync::Arc::new(UsageRouteProvider));
        let value = post_json(
            app,
            "/v1/messages",
            serde_json::json!({
                "model": "mock",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "usage please"}]
            }),
        )
        .await;

        assert_eq!(value["usage"]["input_tokens"], 289);
        assert_eq!(value["usage"]["output_tokens"], 391);
    }

    #[test]
    fn telemetry_record_includes_output_tokens_and_detail_counts() {
        let context = TelemetryContext {
            route: "openai.chat_completions",
            model: "mock".to_string(),
            source_format: "openai_chat",
            tenant_id: "tenant-a".to_string(),
            request_id: "request-a".to_string(),
            conversation_fingerprint: "conversation-a".to_string(),
            reasoning_effort: "high",
            stream: false,
            batch_index: None,
        };
        let output = KernelOutput {
            final_text: "final answer".to_string(),
            task_graph: TaskGraph::new(TaskId::from("root")),
            verification: VerificationReport {
                request_id: RequestId::from("request-a"),
                passed: true,
                issues: vec![],
                artifact_coverage: vec![],
                unresolved_tool_calls: vec![],
                budget_summary: BudgetSummary {
                    token_budget: 4096,
                    token_used: 40,
                    tool_call_budget: 16,
                    tool_calls_used: 0,
                },
            },
            tool_calls: vec![],
            encrypted_subagent_state: vec![],
            usage: ProviderUsage {
                input_tokens: 17,
                output_tokens: 23,
            },
            provider_call_count: 3,
            trace_events: vec![],
        };

        let record = kernel_telemetry_record(&context, &output, None);

        assert_eq!(record["event"], "api_usage");
        assert!(record["timestamp_ms"].as_u64().unwrap_or_default() > 0);
        assert_eq!(record["output_tokens"], 23);
        assert_eq!(record["usage"]["total_tokens"], 40);
        assert_eq!(record["provider_call_count"], 3);
        assert_eq!(record["verification"]["budget_summary"]["token_used"], 40);
        assert_eq!(record["direct_passthrough"], false);
        assert!(record["context"].is_null());
    }

    #[test]
    fn public_reasoning_summary_formats_clean_agent_output_lines() {
        let output = KernelOutput {
            final_text: "final answer".to_string(),
            task_graph: TaskGraph::new(TaskId::from("root")),
            verification: VerificationReport {
                request_id: RequestId::from("request-a"),
                passed: true,
                issues: vec![],
                artifact_coverage: vec![],
                unresolved_tool_calls: vec![],
                budget_summary: BudgetSummary {
                    token_budget: 4096,
                    token_used: 40,
                    tool_call_budget: 16,
                    tool_calls_used: 0,
                },
            },
            tool_calls: vec![],
            encrypted_subagent_state: vec![],
            usage: ProviderUsage {
                input_tokens: 17,
                output_tokens: 23,
            },
            provider_call_count: 2,
            trace_events: vec![
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("deterministic-child-01"),
                    role: AgentRole::Worker,
                    text_outputs: vec![
                        serde_json::json!({
                            "summary": "確認測試完成",
                            "confidence": "high"
                        })
                        .to_string(),
                    ],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 100,
                        output_tokens: 20,
                    },
                },
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("deterministic-child-02"),
                    role: AgentRole::Worker,
                    text_outputs: vec!["測試完成。".to_string()],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 90,
                        output_tokens: 10,
                    },
                },
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("deterministic-child-03"),
                    role: AgentRole::Worker,
                    text_outputs: vec![
                        r#"{ "intent": "確認測試完成", "constraints": ["#.to_string(),
                    ],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 90,
                        output_tokens: 10,
                    },
                },
            ],
        };

        let reasoning = public_reasoning_summary(&output);

        assert!(reasoning.contains(
            "  - deterministic-child-01 (Worker agent): summary: 確認測試完成; confidence: high"
        ));
        assert!(reasoning.contains("  - deterministic-child-02 (Worker agent): 測試完成。"));
        assert!(
            reasoning.contains("  - deterministic-child-03 (Worker agent): intent: 確認測試完成")
        );
        assert!(!reasoning.contains("[no internal tool calls"));
        assert!(!reasoning.contains("tokens in/out"));
        assert!(!reasoning.contains("{\"summary\""));
        assert!(!reasoning.contains(r#"{ "intent""#));
    }

    #[test]
    fn training_example_formats_inputs_tools_middle_trace_and_final_answer() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [
                {"role": "system", "content": "Use concise answers."},
                {"role": "user", "content": "Find the value."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup a value",
                    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}
                }
            }]
        }))
        .unwrap();
        let mut normalized = normalize_openai_chat(request).unwrap();
        let scope = normalized.isolation_key();
        normalized.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-lookup"),
            scope: scope.clone(),
            result_json: serde_json::json!({"value": "42"}),
            result_sha256: "sha".to_string(),
            status: ToolResultStatus::Accepted,
        }];
        let child_task = SubtaskSpec {
            task_id: TaskId::from("child-lookup"),
            parent_task_id: Some(TaskId::from("root")),
            spawn_depth: 1,
            role: AgentRole::Worker,
            objective: "inspect supporting context".to_string(),
            input_artifact_refs: vec![],
            expected_outputs: vec![ArtifactKind::Text],
            allowed_capabilities: CapabilitySet::from([Capability::Text]),
            limits: AgentLimits::default(),
        };
        let output = KernelOutput {
            final_text: "The value is 42.".to_string(),
            task_graph: TaskGraph::new(TaskId::from("root")),
            verification: VerificationReport {
                request_id: normalized.request_id.clone(),
                passed: true,
                issues: vec![],
                artifact_coverage: vec![],
                unresolved_tool_calls: vec![],
                budget_summary: BudgetSummary::default(),
            },
            tool_calls: vec![ToolCallRecord {
                tool_call_id: ToolCallId::from("call-lookup"),
                scope,
                task_id: TaskId::from("root"),
                agent_id: AgentId::from("agent-root"),
                tool_name: "lookup".to_string(),
                arguments_json: serde_json::json!({"key": "answer"}),
                arguments_sha256: "sha".to_string(),
                status: ToolCallStatus::Pending,
                created_at_ms: 1,
                resolved_at_ms: None,
            }],
            encrypted_subagent_state: vec![],
            usage: ProviderUsage::default(),
            provider_call_count: 3,
            trace_events: vec![
                KernelTraceEvent::SpawnPlan {
                    task_id: TaskId::from("root"),
                    reason: "split lookup work".to_string(),
                    children: vec![child_task],
                },
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("child-lookup"),
                    role: AgentRole::Worker,
                    text_outputs: vec!["child observation: supporting context says 42".to_string()],
                    tool_calls: vec![],
                    usage: ProviderUsage::default(),
                },
            ],
        };

        let example = training_example_from_kernel_output(&normalized, &output);
        let conversations = example["conversations"].as_array().unwrap();

        assert_eq!(example["system"], "Use concise answers.");
        assert!(example["tools"].as_str().unwrap().contains("lookup"));
        assert_eq!(conversations.first().unwrap()["from"], "human");
        assert!(
            conversations.first().unwrap()["value"]
                .as_str()
                .unwrap()
                .contains("Find the value.")
        );
        assert!(
            conversations
                .iter()
                .any(|turn| turn["from"] == "function_call"
                    && turn["value"].as_str().unwrap().contains("spawn_agent"))
        );
        assert!(conversations.iter().any(|turn| {
            turn["from"] == "observation"
                && turn["value"]
                    .as_str()
                    .unwrap()
                    .contains("supporting context says 42")
        }));
        assert!(
            conversations
                .iter()
                .any(|turn| turn["from"] == "function_call"
                    && turn["value"].as_str().unwrap().contains("lookup"))
        );
        assert!(
            conversations
                .iter()
                .any(|turn| turn["from"] == "observation"
                    && turn["value"].as_str().unwrap().contains("42"))
        );
        assert_eq!(conversations.last().unwrap()["from"], "gpt");
        assert_eq!(conversations.last().unwrap()["value"], "The value is 42.");
    }

    #[tokio::test]
    async fn routes_openai_training_trace_writes_jsonl_when_enabled() {
        let path =
            std::env::temp_dir().join(format!("miya-training-trace-{}.jsonl", Uuid::new_v4()));
        let app = build_router_with_state(
            AppState::with_provider_context_direct_tenant_limiter_and_training_trace(
                std::sync::Arc::new(UsageRouteProvider),
                ApiContextManager::disabled(),
                DirectBackend::Mock,
                TenantConcurrencyLimiter::disabled(),
                TrainingTraceRecorder::enabled_at(path.clone()),
            ),
        );

        let _ = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "messages": [{"role": "user", "content": "record this"}]
            }),
        )
        .await;

        let content = std::fs::read_to_string(&path).unwrap();
        let line = content.lines().next().unwrap();
        let example: serde_json::Value = serde_json::from_str(line).unwrap();
        let conversations = example["conversations"].as_array().unwrap();
        assert_eq!(conversations[0]["from"], "human");
        assert_eq!(conversations[0]["value"], "record this");
        assert!(conversations.iter().any(|turn| {
            turn["from"] == "function_call"
                && turn["value"].as_str().unwrap().contains("spawn_agent")
        }));
        assert_eq!(conversations.last().unwrap()["from"], "gpt");
        assert_eq!(conversations.last().unwrap()["value"], "usage aware answer");

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn routes_anthropic_messages_returns_tool_use_shape() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "tools": [{
                            "name": "lookup",
                            "input_schema": {"type": "object"}
                        }],
                        "messages": [{"role": "user", "content": "please use a tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["stop_reason"], "tool_use");
        assert_eq!(value["content"][0]["type"], "tool_use");
        assert_eq!(value["content"][0]["name"], "lookup");
    }

    #[tokio::test]
    async fn routes_anthropic_stream_reasoning_effort_none_bypasses_kernel() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "reasoning": {"effort": "none"},
                        "messages": [{"role": "user", "content": "direct anthropic stream"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("event: message_start"));
        assert!(body.contains("direct backend response: direct anthropic stream"));
        assert_eq!(provider.max_in_flight(), 0);
    }

    #[tokio::test]
    async fn routes_openai_stream_true_returns_sse_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("\"object\":\"chat.completion.chunk\""));
        assert!(body.contains("\"delta\":{\"content\":\"Here is a clear, usable answer: hello\"}"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn routes_openai_stream_accepts_root_chat_completions_alias() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "stream alias"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    }

    #[tokio::test]
    async fn routes_openai_stream_accepts_double_v1_chat_completions_alias() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "stream double v1 alias"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    }

    #[tokio::test]
    async fn routes_openai_completions_accepts_sillytavern_generic_stream() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "prompt": "generic prompt",
                        "stream": true,
                        "max_tokens": 65536,
                        "temperature": 0.75,
                        "top_p": 1,
                        "stop": ["<turn|>", "\n\n`Story So Far`:\n"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("\"object\":\"text_completion\""));
        assert!(body.contains("\"choices\":[{\"finish_reason\":null,\"index\":0,\"logprobs\":null,\"text\":\"Here is a clear, usable answer: generic prompt\"}]"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn routes_openai_completions_gemma_reasoning_wraps_public_summary_in_channels() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "local-gemma-finetune",
                        "prompt": "spawn visual inspection",
                        "stream": true,
                        "include_reasoning": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("<|channel>thought"));
        assert!(body.contains("Multi-agent process summary"));
        assert!(body.contains("scheduled 16 bounded child agent"));
        assert!(body.contains("<channel|>"));
        assert!(body.contains("based on the verified agent results"));
        assert!(!body.contains("spawn_plan"));
    }

    #[tokio::test]
    async fn routes_openai_completions_qwen_reasoning_uses_think_tags() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "local-qwen-model",
                        "prompt": "spawn visual inspection",
                        "include_reasoning": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let text = value["choices"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("<think>\nMulti-agent process summary:"));
        assert!(text.contains("\n</think>\n\nHere is a clear, usable answer"));
        assert!(!text.contains("<|channel>thought"));
    }

    #[tokio::test]
    async fn routes_public_reasoning_mode_never_suppresses_requested_summary() {
        let state = AppState {
            public_reasoning_mode: PublicReasoningMode::Never,
            ..AppState::default()
        };
        let app = build_router_with_state(state);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "local-qwen-model",
                        "prompt": "spawn visual inspection",
                        "include_reasoning": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let text = value["choices"][0]["text"].as_str().unwrap();
        assert!(text.contains("based on the verified agent results"));
        assert!(!text.contains("<think>"));
        assert!(!text.contains("Multi-agent process summary"));
    }

    #[tokio::test]
    async fn routes_public_reasoning_env_default_includes_summary_without_request_flag() {
        let state = AppState {
            public_reasoning_mode: PublicReasoningMode::from_env_value(None),
            ..AppState::default()
        };
        let app = build_router_with_state(state);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "local-qwen-model",
                        "prompt": "spawn visual inspection"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let text = value["choices"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("<think>\nMulti-agent process summary:"));
        assert!(text.contains("\n</think>\n\nHere is a clear, usable answer"));
    }

    #[tokio::test]
    async fn routes_openai_completions_accepts_root_and_double_v1_aliases() {
        for uri in ["/completions", "/v1/v1/completions"] {
            let app = build_router_with_provider(std::sync::Arc::new(InputEchoProvider));
            let response = tower::ServiceExt::oneshot(
                app,
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": "mock",
                            "prompt": "legacy completion alias"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

            assert_eq!(response.status(), axum::http::StatusCode::OK, "{uri}");
            let value = response_json(response).await;
            assert_eq!(value["object"], "text_completion", "{uri}");
            assert!(
                value["choices"][0]["text"]
                    .as_str()
                    .unwrap()
                    .contains("legacy completion alias"),
                "{uri}"
            );
        }
    }

    #[tokio::test]
    async fn routes_openai_completions_preserves_generic_model_options() {
        let provider = std::sync::Arc::new(ProviderOptionsProbe::default());
        let app = build_router_with_provider(provider.clone());
        let value = post_json(
            app,
            "/v1/completions",
            serde_json::json!({
                "model": "mock",
                "prompt": "provider options",
                "temperature": 0.75,
                "top_p": 1,
                "max_tokens": 65536,
                "stop": ["<turn|>", "\n\n`Story So Far`:\n"],
                "seed": 123
            }),
        )
        .await;

        assert_eq!(value["object"], "text_completion");
        let options = provider.provider_options();
        assert_eq!(options["temperature"], 0.75);
        assert_eq!(options["top_p"], 1);
        assert_eq!(options["max_tokens"], 65536);
        assert_eq!(
            options["stop"],
            serde_json::json!(["<turn|>", "\n\n`Story So Far`:\n"])
        );
        assert_eq!(options["seed"], 123);
        assert!(options.get("prompt").is_none());
    }

    #[tokio::test]
    async fn routes_openai_stream_uses_provider_token_stream_before_completion() {
        let provider = std::sync::Arc::new(StreamingProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(provider.invoke_calls(), 0);
        assert_eq!(provider.stream_calls(), 1);
        let first_text = first_stream_text(response).await;
        assert!(first_text.contains("stream-token-1"));
    }

    #[tokio::test]
    async fn routes_openai_stream_tool_call_returns_sse_tool_delta() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "please use a tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("\"tool_calls\""));
        assert!(body.contains("\"finish_reason\":\"tool_calls\""));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn routes_openai_stream_tool_delta_overrides_stop_finish_reason() {
        let app = build_router_with_provider(std::sync::Arc::new(StopAfterToolStreamProvider));
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("\"tool_calls\""));
        assert!(body.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[tokio::test]
    async fn routes_openai_stream_spawn_uses_orchestrated_final_synthesis() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("based on the verified agent results"));
        assert!(!body.contains("spawn_plan"));
        assert!(!body.contains("child visual inspection"));
    }

    #[tokio::test]
    async fn routes_openai_chat_completion_includes_public_reasoning_summary_when_requested() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "custom-model",
                        "include_reasoning": true,
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let message = &value["choices"][0]["message"];
        assert!(
            message["content"]
                .as_str()
                .unwrap()
                .contains("based on the verified agent results")
        );
        let reasoning = message["reasoning_content"].as_str().unwrap();
        assert!(reasoning.contains("Multi-agent process summary"));
        assert!(reasoning.contains("scheduled 16 bounded child agent"));
        assert!(reasoning.contains("Agent output summaries:"));
        assert!(
            reasoning.contains("deterministic-child-01 (Worker agent): summarized worker finding")
        );
        assert!(!reasoning.contains("child completed:"));
        assert!(!reasoning.contains("returned 1 text artifact(s)"));
        assert!(reasoning.contains("Verification passed"));
        assert_eq!(message["reasoning"]["summary"], reasoning);
    }

    #[tokio::test]
    async fn routes_openai_stream_include_reasoning_uses_summary_delta() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "custom-model",
                        "stream": true,
                        "include_reasoning": true,
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("\"reasoning_content\":\"Multi-agent process summary"));
        assert!(body.contains(
            "\"content\":\"Here is a clear, usable answer based on the verified agent results.\""
        ));
        assert!(body.contains("Agent output summaries:"));
        assert!(body.contains("deterministic-child-01 (Worker agent): summarized worker finding"));
        assert!(!body.contains("child completed:"));
    }

    #[tokio::test]
    async fn routes_openai_stream_include_reasoning_forces_orchestrated_summary() {
        let provider = std::sync::Arc::new(StreamingProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "include_reasoning": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(provider.invoke_calls(), 18);
        assert_eq!(provider.stream_calls(), 0);
        assert!(body.contains("\"reasoning_content\":\"Multi-agent process summary"));
        assert!(body.contains("scheduled 16 bounded child agent"));
        assert!(body.contains("\"content\":\"non-stream fallback\""));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_true_returns_sse_answer() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("\"text\":\"Here is a clear, usable answer: hello\""));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_uses_provider_token_stream_before_completion() {
        let provider = std::sync::Arc::new(StreamingProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(provider.invoke_calls(), 0);
        assert_eq!(provider.stream_calls(), 1);
        let first_text = first_stream_text(response).await;
        assert!(first_text.contains("stream-token-1"));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_include_reasoning_forces_thinking_summary_block() {
        let provider = std::sync::Arc::new(StreamingProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "include_reasoning": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(provider.invoke_calls(), 18);
        assert_eq!(provider.stream_calls(), 0);
        assert!(body.contains("\"type\":\"thinking_delta\""));
        assert!(body.contains("Multi-agent process summary"));
        assert!(body.contains("\"text\":\"non-stream fallback\""));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_tool_use_returns_sse_tool_use() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "messages": [{"role": "user", "content": "please use a tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("event: content_block_start"));
        assert!(body.contains("\"type\":\"tool_use\""));
        assert!(body.contains("\"stop_reason\":\"tool_use\""));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_tool_delta_overrides_stop_reason() {
        let app = build_router_with_provider(std::sync::Arc::new(StopAfterToolStreamProvider));
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "messages": [{"role": "user", "content": "tool"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("\"type\":\"tool_use\""));
        assert!(body.contains("\"stop_reason\":\"tool_use\""));
    }

    #[tokio::test]
    async fn routes_anthropic_stream_spawn_uses_orchestrated_final_synthesis() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "stream": true,
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("based on the verified agent results"));
        assert!(!body.contains("spawn_plan"));
        assert!(!body.contains("child visual inspection"));
    }

    #[tokio::test]
    async fn routes_do_not_expose_subagent_state_by_default() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert!(value.get("encrypted_agent_state").is_none());
        let wire = value.to_string();
        assert!(!wire.contains("child completed"));
        assert!(!wire.contains("child visual inspection"));
    }

    #[tokio::test]
    async fn routes_include_encrypted_subagent_state_when_requested() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "metadata": {"include_encrypted_subagent_state": true},
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let encrypted = value["encrypted_agent_state"].as_array().unwrap();
        assert_eq!(encrypted.len(), 16);
        assert_eq!(encrypted[0]["algorithm"], "AES-256-GCM");
        let wire = value.to_string();
        assert!(!wire.contains("child completed"));
    }

    #[tokio::test]
    async fn routes_anthropic_include_encrypted_subagent_state_when_requested() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "max_tokens": 256,
                        "metadata": {"include_encrypted_subagent_state": true},
                        "messages": [{"role": "user", "content": "spawn visual inspection"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(
            value["encrypted_agent_state"][0]["algorithm"],
            "AES-256-GCM"
        );
        let wire = value.to_string();
        assert!(!wire.contains("child completed"));
        assert!(!wire.contains("child visual inspection"));
    }

    #[tokio::test]
    async fn routes_openai_batch_keeps_outputs_isolated_by_request() {
        let app = build_router_with_provider(std::sync::Arc::new(InputEchoProvider));
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"model": "mock", "messages": [{"role": "user", "content": "alpha conversation"}]},
                            {"model": "mock", "messages": [{"role": "user", "content": "beta conversation"}]}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        let first = value["data"][0]["response"]["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        let second = value["data"][1]["response"]["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert!(first.contains("alpha conversation"));
        assert!(!first.contains("beta conversation"));
        assert!(second.contains("beta conversation"));
        assert!(!second.contains("alpha conversation"));
    }

    #[tokio::test]
    async fn routes_openai_batch_runs_requests_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"model": "mock", "messages": [{"role": "user", "content": "first"}]},
                            {"model": "mock", "messages": [{"role": "user", "content": "second"}]}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(provider.max_in_flight() >= 2);
    }

    #[tokio::test]
    async fn routes_openai_batch_limits_same_tenant_concurrency() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {
                                "model": "mock",
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "first"}]
                            },
                            {
                                "model": "mock",
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "second"}]
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(provider.max_in_flight(), 4);
    }

    #[tokio::test]
    async fn routes_openai_batch_allows_different_tenants_to_run_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {
                                "model": "mock",
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "first"}]
                            },
                            {
                                "model": "mock",
                                "metadata": {"tenant_id": "tenant-b"},
                                "messages": [{"role": "user", "content": "second"}]
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(provider.max_in_flight() >= 2);
    }

    #[tokio::test]
    async fn routes_reasoning_effort_improves_eval_coverage() {
        let app = build_router_with_provider(std::sync::Arc::new(EffortCoverageRouteProvider));

        let low = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "reasoning": {"effort": "low"},
                "metadata": {"include_encrypted_subagent_state": true},
                "messages": [{"role": "user", "content": "deep compare across eight independent dimensions"}]
            }),
        )
        .await;
        let high = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "reasoning": {"effort": "high"},
                "metadata": {
                    "include_encrypted_subagent_state": true,
                    "agent": {"max_parallel_agents": 8}
                },
                "messages": [{"role": "user", "content": "deep compare across eight independent dimensions"}]
            }),
        )
        .await;

        let low_score = coverage_score_from_openai_response(&low);
        let high_score = coverage_score_from_openai_response(&high);
        assert_eq!(low_score, 4);
        assert_eq!(high_score, 32);
        assert!(high_score > low_score);
        assert_eq!(low["encrypted_agent_state"].as_array().unwrap().len(), 4);
        assert_eq!(high["encrypted_agent_state"].as_array().unwrap().len(), 32);
    }

    #[tokio::test]
    async fn routes_openai_context_retrieves_surreal_kv_memory() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let first = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {"id": "long-context-memory", "include_report": true}
                },
                "messages": [{"role": "user", "content": "remember needle marker alpha-771"}]
            }),
        )
        .await;
        assert_eq!(first["context_cache"]["stored_revision"], 2);

        let second = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "long-context-memory",
                        "query": "alpha-771",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "what marker is in memory?"}]
            }),
        )
        .await;

        let content = second["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("alpha-771"));
        assert_eq!(second["context_cache"]["enabled"], true);
        assert!(second["context_cache"]["included_bytes"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn routes_openai_context_cache_reuses_common_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let _ = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {"context": {"id": "cache-memory"}},
                "messages": [{"role": "user", "content": "common lore cache-key amber"}]
            }),
        )
        .await;

        let second = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "cache-memory",
                        "query": "amber",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "retrieve amber"}]
            }),
        )
        .await;
        assert_eq!(second["context_cache"]["cache_hit"], false);

        let third = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "cache-memory",
                        "query": "amber",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "retrieve amber again"}]
            }),
        )
        .await;
        assert_eq!(third["context_cache"]["cache_hit"], true);
        assert!(
            third["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("amber")
        );
    }

    #[tokio::test]
    async fn routes_openai_context_cache_isolates_model_and_explicit_namespace() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let _ = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {"context": {"id": "isolated-cache-memory"}},
                "messages": [{"role": "user", "content": "shared context marker sapphire"}]
            }),
        )
        .await;

        let mock_warm = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "isolated-cache-memory",
                        "query": "sapphire",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "retrieve sapphire"}]
            }),
        )
        .await;
        assert_eq!(mock_warm["context_cache"]["cache_hit"], false);

        let other_model = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "other-backend-model",
                "metadata": {
                    "context": {
                        "id": "isolated-cache-memory",
                        "query": "sapphire",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "retrieve sapphire on other model"}]
            }),
        )
        .await;
        assert_eq!(other_model["context_cache"]["cache_hit"], false);

        let explicit_namespace = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "other-backend-model",
                "metadata": {
                    "context": {
                        "id": "isolated-cache-memory",
                        "query": "sapphire",
                        "append": false,
                        "include_report": true,
                        "cache_namespace": "manual-eval-profile"
                    }
                },
                "messages": [{"role": "user", "content": "retrieve sapphire in manual namespace"}]
            }),
        )
        .await;
        assert_eq!(explicit_namespace["context_cache"]["cache_hit"], false);

        let explicit_namespace_again = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "other-backend-model",
                "metadata": {
                    "context": {
                        "id": "isolated-cache-memory",
                        "query": "sapphire",
                        "append": false,
                        "include_report": true,
                        "cache_namespace": "manual-eval-profile"
                    }
                },
                "messages": [{"role": "user", "content": "retrieve sapphire in manual namespace again"}]
            }),
        )
        .await;
        assert_eq!(explicit_namespace_again["context_cache"]["cache_hit"], true);
    }

    #[tokio::test]
    async fn routes_anthropic_batch_returns_isolated_responses() {
        let app = build_router_with_provider(std::sync::Arc::new(InputEchoProvider));
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "alpha anthropic"}]},
                            {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "beta anthropic"}]}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let value = response_json(response).await;
        assert_eq!(value["type"], "message_batch");
        let first = value["data"][0]["response"]["content"][0]["text"]
            .as_str()
            .unwrap();
        let second = value["data"][1]["response"]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert!(first.contains("alpha anthropic"));
        assert!(!first.contains("beta anthropic"));
        assert!(second.contains("beta anthropic"));
        assert!(!second.contains("alpha anthropic"));
    }

    #[tokio::test]
    async fn routes_anthropic_batch_runs_requests_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "first anthropic"}]},
                            {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "second anthropic"}]}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(provider.max_in_flight() >= 2);
    }

    #[tokio::test]
    async fn routes_anthropic_batch_limits_same_tenant_concurrency() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {
                                "model": "mock",
                                "max_tokens": 256,
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "first anthropic"}]
                            },
                            {
                                "model": "mock",
                                "max_tokens": 256,
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "second anthropic"}]
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(provider.max_in_flight(), 4);
    }

    #[tokio::test]
    async fn routes_anthropic_batch_allows_different_tenants_to_run_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batch")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {
                                "model": "mock",
                                "max_tokens": 256,
                                "metadata": {"tenant_id": "tenant-a"},
                                "messages": [{"role": "user", "content": "first anthropic"}]
                            },
                            {
                                "model": "mock",
                                "max_tokens": 256,
                                "metadata": {"tenant_id": "tenant-b"},
                                "messages": [{"role": "user", "content": "second anthropic"}]
                            }
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(provider.max_in_flight() >= 2);
    }

    #[tokio::test]
    async fn routes_anthropic_context_retrieves_surreal_kv_memory() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let _ = post_json(
            app.clone(),
            "/v1/messages",
            serde_json::json!({
                "model": "mock",
                "max_tokens": 256,
                "metadata": {"context": {"id": "anthropic-memory"}},
                "messages": [{"role": "user", "content": "remember anthropic needle teal"}]
            }),
        )
        .await;

        let second = post_json(
            app,
            "/v1/messages",
            serde_json::json!({
                "model": "mock",
                "max_tokens": 256,
                "metadata": {
                    "context": {
                        "id": "anthropic-memory",
                        "query": "teal",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "what color is stored?"}]
            }),
        )
        .await;

        assert!(
            second["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("teal")
        );
        assert_eq!(second["context_cache"]["enabled"], true);
    }

    async fn post_json(app: axum::Router, uri: &str, body: serde_json::Value) -> serde_json::Value {
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        response_json(response).await
    }

    async fn post_json_with_header(
        app: axum::Router,
        uri: &str,
        header_name: &str,
        header_value: &str,
        body: serde_json::Value,
    ) -> serde_json::Value {
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .header(header_name, header_value)
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        response_json(response).await
    }

    #[tokio::test]
    async fn routes_openai_context_isolated_by_header_tenant() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let _ = post_json_with_header(
            app.clone(),
            "/v1/chat/completions",
            "x-tenant-id",
            "tenant-a",
            serde_json::json!({
                "model": "mock",
                "metadata": {"context": {"id": "shared-memory"}},
                "messages": [{"role": "user", "content": "tenant-a-only redwood marker"}]
            }),
        )
        .await;

        let tenant_b = post_json_with_header(
            app.clone(),
            "/v1/chat/completions",
            "x-tenant-id",
            "tenant-b",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "shared-memory",
                        "query": "redwood",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "what marker is in memory?"}]
            }),
        )
        .await;
        assert_eq!(tenant_b["context_cache"]["included_bytes"], 0);
        assert!(
            !tenant_b["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("redwood")
        );

        let tenant_a = post_json_with_header(
            app,
            "/v1/chat/completions",
            "x-tenant-id",
            "tenant-a",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {
                        "id": "shared-memory",
                        "query": "redwood",
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "what marker is in memory?"}]
            }),
        )
        .await;
        assert!(
            tenant_a["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("redwood")
        );
    }

    #[tokio::test]
    async fn routes_openai_batch_context_isolated_by_item_tenant() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(InputEchoProvider), context);

        let _ = post_json(
            app.clone(),
            "/v1/chat/completions/batch",
            serde_json::json!({
                "requests": [
                    {
                        "model": "mock",
                        "metadata": {
                            "tenant_id": "batch-tenant-a",
                            "context": {"id": "batch-shared-memory"}
                        },
                        "messages": [{"role": "user", "content": "batch tenant a silver-needle"}]
                    },
                    {
                        "model": "mock",
                        "metadata": {
                            "tenant_id": "batch-tenant-b",
                            "context": {"id": "batch-shared-memory"}
                        },
                        "messages": [{"role": "user", "content": "batch tenant b copper-marker"}]
                    }
                ]
            }),
        )
        .await;

        let tenant_b = post_json(
            app,
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "tenant_id": "batch-tenant-b",
                    "context": {
                        "id": "batch-shared-memory",
                        "query": "silver-needle",
                        "recent_tail_chunks": 0,
                        "append": false,
                        "include_report": true
                    }
                },
                "messages": [{"role": "user", "content": "retrieve silver needle"}]
            }),
        )
        .await;

        assert_eq!(tenant_b["context_cache"]["included_bytes"], 0);
        assert!(
            !tenant_b["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("silver-needle")
        );
    }

    #[derive(Debug)]
    struct InputEchoProvider;

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for InputEchoProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: format!("echo {}", request.input_text),
                }],
                tool_calls: Vec::new(),
                usage: ProviderUsage::default(),
            })
        }
    }

    #[derive(Debug, Clone)]
    struct UsageRouteProvider;

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for UsageRouteProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: "usage aware answer".to_string(),
                }],
                tool_calls: Vec::new(),
                usage: ProviderUsage {
                    input_tokens: 17,
                    output_tokens: 23,
                },
            })
        }
    }

    #[derive(Debug, Default)]
    struct ProviderOptionsProbe {
        provider_options: std::sync::Mutex<Option<serde_json::Value>>,
    }

    impl ProviderOptionsProbe {
        fn provider_options(&self) -> serde_json::Value {
            self.provider_options
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_default()
        }
    }

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for ProviderOptionsProbe {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            *self.provider_options.lock().unwrap() = Some(request.provider_options);
            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: "provider options captured".to_string(),
                }],
                tool_calls: Vec::new(),
                usage: ProviderUsage::default(),
            })
        }
    }

    #[derive(Debug, Default)]
    struct ConcurrentProbeProvider {
        in_flight: std::sync::atomic::AtomicUsize,
        max_in_flight: std::sync::atomic::AtomicUsize,
    }

    impl ConcurrentProbeProvider {
        fn max_in_flight(&self) -> usize {
            self.max_in_flight.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for ConcurrentProbeProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            let active = self
                .in_flight
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.max_in_flight
                .fetch_max(active, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: format!("echo {}", request.input_text),
                }],
                tool_calls: Vec::new(),
                usage: Default::default(),
            })
        }
    }

    struct EffortCoverageRouteProvider;

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for EffortCoverageRouteProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            match request.task.role {
                AgentRole::Leader => {
                    let target = route_target_parallel_agents(&request.system_instructions).min(8);
                    Ok(provider_core::ProviderResponse {
                        artifacts: vec![AgentArtifact::SpawnPlan {
                            id: ArtifactId::from("spawn-plan-effort-route"),
                            scope: request.scope.clone(),
                            plan: SpawnPlan {
                                parent_task_id: request.task.task_id,
                                reason: "cover independent route evaluation dimensions".to_string(),
                                children: (0..target)
                                    .map(|index| SubtaskSpec {
                                        task_id: TaskId::from(format!("route-coverage-{index}")),
                                        parent_task_id: Some(TaskId::from("root")),
                                        spawn_depth: 1,
                                        role: AgentRole::Worker,
                                        objective: format!("cover route dimension {index}"),
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
                AgentRole::Worker => Ok(provider_core::ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                        scope: request.scope,
                        text: format!("covered route dimension: {}", request.task.objective),
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::ReasoningSummarizer => Ok(provider_core::ProviderResponse {
                    artifacts: vec![AgentArtifact::Text {
                        id: ArtifactId::from("route-coverage-summary"),
                        scope: request.scope,
                        text: request.input_text,
                    }],
                    tool_calls: vec![],
                    usage: Default::default(),
                }),
                AgentRole::Synthesizer => {
                    let score = request
                        .input_text
                        .matches("covered route dimension:")
                        .count();
                    Ok(provider_core::ProviderResponse {
                        artifacts: vec![AgentArtifact::Text {
                            id: ArtifactId::from("route-coverage-score"),
                            scope: request.scope,
                            text: format!("coverage_score={score}"),
                        }],
                        tool_calls: vec![],
                        usage: Default::default(),
                    })
                }
                AgentRole::Verifier => Ok(provider_core::ProviderResponse::default()),
            }
        }
    }

    fn route_target_parallel_agents(instructions: &[String]) -> usize {
        instructions
            .iter()
            .flat_map(|instruction| instruction.split([';', '\n']))
            .filter_map(|part| part.trim().strip_prefix("target_parallel_agents="))
            .filter_map(|value| value.parse::<usize>().ok())
            .next()
            .unwrap_or(1)
    }

    fn coverage_score_from_openai_response(value: &serde_json::Value) -> usize {
        value["choices"][0]["message"]["content"]
            .as_str()
            .and_then(|content| content.strip_prefix("coverage_score="))
            .and_then(|score| score.parse::<usize>().ok())
            .unwrap()
    }

    #[derive(Debug, Default)]
    struct StreamingProbeProvider {
        invoke_calls: std::sync::atomic::AtomicUsize,
        stream_calls: std::sync::atomic::AtomicUsize,
    }

    impl StreamingProbeProvider {
        fn invoke_calls(&self) -> usize {
            self.invoke_calls.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn stream_calls(&self) -> usize {
            self.stream_calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for StreamingProbeProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            self.invoke_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from(format!("text-{}", request.task.task_id.as_ref())),
                    scope: request.scope,
                    text: "non-stream fallback".to_string(),
                }],
                tool_calls: Vec::new(),
                usage: Default::default(),
            })
        }

        async fn stream(
            &self,
            _request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderStream, provider_core::ProviderError> {
            self.stream_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Box::pin(futures::stream::unfold(0_u8, |step| async move {
                match step {
                    0 => Some((
                        Ok(provider_core::ProviderStreamEvent::TextDelta {
                            text: "stream-token-1".to_string(),
                        }),
                        1,
                    )),
                    1 => {
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        Some((
                            Ok(provider_core::ProviderStreamEvent::Finish {
                                reason: provider_core::ProviderFinishReason::Stop,
                            }),
                            2,
                        ))
                    }
                    _ => None,
                }
            })))
        }
    }

    struct StopAfterToolStreamProvider;

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for StopAfterToolStreamProvider {
        async fn invoke(
            &self,
            _request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            panic!("streaming route must not call invoke")
        }

        async fn stream(
            &self,
            _request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderStream, provider_core::ProviderError> {
            Ok(Box::pin(futures::stream::iter([
                Ok(provider_core::ProviderStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some(ToolCallId::from("call-1")),
                    name: Some("lookup".to_string()),
                    arguments_delta: "{\"query\":\"x\"}".to_string(),
                }),
                Ok(provider_core::ProviderStreamEvent::Finish {
                    reason: provider_core::ProviderFinishReason::Stop,
                }),
            ])))
        }
    }

    async fn first_stream_text(response: axum::response::Response) -> String {
        use futures::StreamExt;

        let mut stream = response.into_body().into_data_stream();
        let mut body = String::new();
        for _ in 0..4 {
            let chunk = tokio::time::timeout(std::time::Duration::from_millis(200), stream.next())
                .await
                .expect("stream did not yield early")
                .expect("stream ended before text")
                .expect("stream chunk failed");
            body.push_str(std::str::from_utf8(&chunk).unwrap());
            if body.contains("stream-token-1") {
                return body;
            }
        }
        body
    }

    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn response_text(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }
}
