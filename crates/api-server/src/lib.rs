mod batch_api;
mod durable;
mod observability;

use batch_api::*;
pub use observability::ObservabilityGuard;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    convert::Infallible,
    fs::{self, OpenOptions},
    future::Future,
    io::Write,
    path::PathBuf,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use agent_kernel::{
    KernelError, KernelOutput, KernelPolicy, KernelRunner, KernelTraceEvent, MockProvider,
};
use agent_protocol::*;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{
        DefaultBodyLimit, FromRequestParts, Multipart, Path, Query, Request, State,
        ws::{Message as AxumWsMessage, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode, header, request::Parts},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, SecondsFormat, Utc};
use context_store::{
    ContextAppendRecord, ContextAssembly, ContextAssemblyOptions, DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_CONTEXT_BYTES, DEFAULT_RECENT_TAIL_CHUNKS, SurrealKvContextStore,
};
use durable::DurableStore;
use futures::{Stream, StreamExt};
use provider_anthropic::AnthropicProvider;
use provider_core::{
    ModelProvider, ProviderError, ProviderFinishReason, ProviderStream, ProviderStreamEvent,
    ProviderUsage,
};
use provider_openai::OpenAiProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use uuid::Uuid;

const MAX_OPENAI_BATCH_REQUESTS: usize = 50_000;
const MAX_ANTHROPIC_BATCH_REQUESTS: usize = 100_000;
const MAX_OPENAI_FILE_BYTES: usize = 512 * 1024 * 1024;
const MAX_OPENAI_BATCH_FILE_BYTES: usize = 200 * 1024 * 1024;
const DEFAULT_TENANT_ID: &str = "default";
const DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS: usize = 16;
const DEFAULT_TENANT_QUEUE_TIMEOUT_MS: u64 = 30_000;
const OPENAI_FILES_NAMESPACE: &str = "openai_files";
const OPENAI_FILE_BLOBS_NAMESPACE: &str = "openai_file_blobs";
const OPENAI_BATCHES_NAMESPACE: &str = "openai_batches";
const ANTHROPIC_BATCHES_NAMESPACE: &str = "anthropic_message_batches";
const ANTHROPIC_BATCH_INPUTS_NAMESPACE: &str = "anthropic_message_batch_inputs";
const ANTHROPIC_BATCH_RESULTS_NAMESPACE: &str = "anthropic_message_batch_results";
const BACKGROUND_RESPONSES_NAMESPACE: &str = "openai_background_responses";
const BACKGROUND_RESPONSE_INPUTS_NAMESPACE: &str = "openai_background_response_inputs";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ApiError {
    #[error("streaming is not supported by the MVP multi-agent kernel")]
    StreamUnsupported,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
pub struct OpenAiResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: serde_json::Value,
    #[serde(default)]
    pub instructions: serde_json::Value,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: serde_json::Value,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub thinking: serde_json::Value,
    #[serde(default)]
    pub reasoning: serde_json::Value,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_logprobs: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub background: bool,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub include: serde_json::Value,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAiContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAiMessageToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<OpenAiLegacyFunctionCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiMessageToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: String,
    pub function: OpenAiMessageToolCallFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiMessageToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiLegacyFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: OpenAiImageUrl,
    },
    InputAudio {
        input_audio: serde_json::Value,
        #[serde(flatten)]
        extra: BTreeMap<String, serde_json::Value>,
    },
    File {
        file: serde_json::Value,
        #[serde(flatten)]
        extra: BTreeMap<String, serde_json::Value>,
    },
    Refusal {
        refusal: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiTool {
    #[serde(default)]
    pub r#type: String,
    pub function: OpenAiFunctionTool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiFunctionTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

#[derive(Clone, Debug)]
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
    ProviderContent {
        value: serde_json::Value,
    },
}

impl<'de> Deserialize<'de> for AnthropicContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let value = serde_json::Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| D::Error::custom("Anthropic content blocks must be objects"))?;
        let kind = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| D::Error::custom("Anthropic content blocks must include type"))?;
        match kind {
            "text"
                if object
                    .keys()
                    .all(|key| matches!(key.as_str(), "type" | "text")) =>
            {
                let text = object
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| D::Error::custom("text block must include text"))?;
                Ok(Self::Text {
                    text: text.to_string(),
                })
            }
            "image"
                if object
                    .keys()
                    .all(|key| matches!(key.as_str(), "type" | "source")) =>
            {
                let source = serde_json::from_value(
                    object
                        .get("source")
                        .cloned()
                        .ok_or_else(|| D::Error::custom("image block must include source"))?,
                )
                .map_err(D::Error::custom)?;
                Ok(Self::Image { source })
            }
            "tool_use"
                if object
                    .keys()
                    .all(|key| matches!(key.as_str(), "type" | "id" | "name" | "input")) =>
            {
                Ok(Self::ToolUse {
                    id: object
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| D::Error::custom("tool_use block must include id"))?
                        .to_string(),
                    name: object
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| D::Error::custom("tool_use block must include name"))?
                        .to_string(),
                    input: object
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                })
            }
            "tool_result"
                if object.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "type" | "tool_use_id" | "content" | "is_error"
                    )
                }) =>
            {
                Ok(Self::ToolResult {
                    tool_use_id: object
                        .get("tool_use_id")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            D::Error::custom("tool_result block must include tool_use_id")
                        })?
                        .to_string(),
                    content: object
                        .get("content")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                })
            }
            _ => Ok(Self::ProviderContent { value }),
        }
    }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredOpenAiFile {
    tenant_id: String,
    id: String,
    bytes: u64,
    created_at: u64,
    filename: String,
    purpose: String,
    status: String,
    expires_at: Option<u64>,
    status_details: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OpenAiBatchRecord {
    tenant_id: String,
    id: String,
    completion_window: String,
    created_at: u64,
    endpoint: String,
    input_file_id: String,
    status: String,
    cancelled_at: Option<u64>,
    cancelling_at: Option<u64>,
    completed_at: Option<u64>,
    error_file_id: Option<String>,
    errors: Option<serde_json::Value>,
    expired_at: Option<u64>,
    expires_at: Option<u64>,
    failed_at: Option<u64>,
    finalizing_at: Option<u64>,
    in_progress_at: Option<u64>,
    metadata: serde_json::Value,
    output_file_id: Option<String>,
    request_counts: OpenAiBatchRequestCounts,
    usage: ProviderUsage,
    cancel_requested: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct OpenAiBatchRequestCounts {
    total: u64,
    completed: u64,
    failed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AnthropicBatchRecord {
    tenant_id: String,
    id: String,
    created_at: u64,
    expires_at: u64,
    ended_at: Option<u64>,
    cancel_initiated_at: Option<u64>,
    archived_at: Option<u64>,
    processing_status: String,
    request_counts: AnthropicBatchRequestCounts,
    cancel_requested: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AnthropicBatchRequestCounts {
    processing: u64,
    succeeded: u64,
    errored: u64,
    canceled: u64,
    expired: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BackgroundResponseJob {
    tenant_id: String,
    response_id: String,
    created_at: u64,
    status: String,
    cancel_requested: bool,
    last_error: Option<String>,
}

#[derive(Clone)]
struct JobRuntime {
    running: Arc<Mutex<HashSet<String>>>,
    cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    semaphore: Arc<Semaphore>,
    metrics: RuntimeMetrics,
}

impl JobRuntime {
    fn new(max_concurrent: usize, metrics: RuntimeMetrics) -> Self {
        Self {
            running: Arc::new(Mutex::new(HashSet::new())),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(max_concurrent.max(1))),
            metrics,
        }
    }

    fn spawn<F, Fut>(&self, key: String, worker: F) -> bool
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut running = match self.running.lock() {
            Ok(running) => running,
            Err(_) => return false,
        };
        if !running.insert(key.clone()) {
            return false;
        }
        drop(running);

        let cancellation = CancellationToken::new();
        if let Ok(mut cancellations) = self.cancellations.lock() {
            cancellations.insert(key.clone(), cancellation.clone());
        }
        let runtime = self.clone();
        let span = tracing::info_span!("durable.job", job.key = %key);
        tokio::spawn(
            async move {
                let permit = runtime.semaphore.clone().acquire_owned().await;
                if permit.is_ok() {
                    runtime.metrics.jobs_started.fetch_add(1, Ordering::Relaxed);
                    worker(cancellation).await;
                    runtime
                        .metrics
                        .jobs_finished
                        .fetch_add(1, Ordering::Relaxed);
                }
                if let Ok(mut cancellations) = runtime.cancellations.lock() {
                    cancellations.remove(&key);
                }
                if let Ok(mut running) = runtime.running.lock() {
                    running.remove(&key);
                }
            }
            .instrument(span),
        );
        true
    }

    fn cancel(&self, key: &str) -> bool {
        let cancellation = self
            .cancellations
            .lock()
            .ok()
            .and_then(|cancellations| cancellations.get(key).cloned());
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
            self.metrics.jobs_cancelled.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Default)]
pub struct RuntimeMetrics {
    http_requests: Arc<AtomicU64>,
    http_failures: Arc<AtomicU64>,
    http_latency_micros: Arc<AtomicU64>,
    orchestration_streams_active: Arc<AtomicU64>,
    orchestration_stream_heartbeats: Arc<AtomicU64>,
    provider_attempts: Arc<AtomicU64>,
    provider_retries: Arc<AtomicU64>,
    provider_failures: Arc<AtomicU64>,
    circuit_rejections: Arc<AtomicU64>,
    jobs_started: Arc<AtomicU64>,
    jobs_finished: Arc<AtomicU64>,
    jobs_cancelled: Arc<AtomicU64>,
}

impl RuntimeMetrics {
    fn prometheus(&self) -> String {
        format!(
            concat!(
                "# TYPE miya_http_requests_total counter\nmiya_http_requests_total {}\n",
                "# TYPE miya_http_failures_total counter\nmiya_http_failures_total {}\n",
                "# TYPE miya_http_request_duration_microseconds_total counter\nmiya_http_request_duration_microseconds_total {}\n",
                "# TYPE miya_orchestration_streams_active gauge\nmiya_orchestration_streams_active {}\n",
                "# TYPE miya_orchestration_stream_heartbeats_total counter\nmiya_orchestration_stream_heartbeats_total {}\n",
                "# TYPE miya_provider_attempts_total counter\nmiya_provider_attempts_total {}\n",
                "# TYPE miya_provider_retries_total counter\nmiya_provider_retries_total {}\n",
                "# TYPE miya_provider_failures_total counter\nmiya_provider_failures_total {}\n",
                "# TYPE miya_provider_circuit_rejections_total counter\nmiya_provider_circuit_rejections_total {}\n",
                "# TYPE miya_durable_jobs_started_total counter\nmiya_durable_jobs_started_total {}\n",
                "# TYPE miya_durable_jobs_finished_total counter\nmiya_durable_jobs_finished_total {}\n",
                "# TYPE miya_durable_jobs_cancelled_total counter\nmiya_durable_jobs_cancelled_total {}\n"
            ),
            self.http_requests.load(Ordering::Relaxed),
            self.http_failures.load(Ordering::Relaxed),
            self.http_latency_micros.load(Ordering::Relaxed),
            self.orchestration_streams_active.load(Ordering::Relaxed),
            self.orchestration_stream_heartbeats.load(Ordering::Relaxed),
            self.provider_attempts.load(Ordering::Relaxed),
            self.provider_retries.load(Ordering::Relaxed),
            self.provider_failures.load(Ordering::Relaxed),
            self.circuit_rejections.load(Ordering::Relaxed),
            self.jobs_started.load(Ordering::Relaxed),
            self.jobs_finished.load(Ordering::Relaxed),
            self.jobs_cancelled.load(Ordering::Relaxed),
        )
    }
}

#[derive(Clone)]
pub struct AppState {
    kernel: Arc<KernelRunner<Arc<dyn ModelProvider>>>,
    context: ApiContextManager,
    responses: ResponsesStore,
    direct: DirectBackend,
    tenant_limiter: TenantConcurrencyLimiter,
    training_trace: TrainingTraceRecorder,
    public_reasoning_mode: PublicReasoningMode,
    shared_api_key: Option<Arc<str>>,
    durable: DurableStore,
    jobs: JobRuntime,
    metrics: RuntimeMetrics,
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

const RESPONSES_STORE_NAMESPACE: &str = "openai_responses";

#[derive(Clone)]
struct ResponsesStore {
    kv: Option<Arc<SurrealKvContextStore>>,
    memory: Arc<Mutex<BTreeMap<String, StoredOpenAiResponse>>>,
}

impl ResponsesStore {
    fn new(kv: Option<Arc<SurrealKvContextStore>>) -> Self {
        Self {
            kv,
            memory: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    async fn put(&self, response: StoredOpenAiResponse) -> Result<(), String> {
        if let Some(kv) = &self.kv {
            kv.put_json(
                RESPONSES_STORE_NAMESPACE,
                &response.tenant_id,
                &response.id,
                serde_json::to_value(&response).map_err(|error| error.to_string())?,
            )
            .await
            .map_err(|error| error.to_string())?;
        }

        let key = response_store_memory_key(&response.tenant_id, &response.id);
        self.memory
            .lock()
            .map_err(|_| "responses store lock poisoned".to_string())?
            .insert(key, response);
        Ok(())
    }

    fn get(
        &self,
        tenant_id: &str,
        response_id: &str,
    ) -> Result<Option<StoredOpenAiResponse>, String> {
        if let Some(kv) = &self.kv
            && let Some(value) = kv
                .get_json(RESPONSES_STORE_NAMESPACE, tenant_id, response_id)
                .map_err(|error| error.to_string())?
        {
            return serde_json::from_value(value)
                .map(Some)
                .map_err(|error| error.to_string());
        }

        let key = response_store_memory_key(tenant_id, response_id);
        self.memory
            .lock()
            .map_err(|_| "responses store lock poisoned".to_string())
            .map(|responses| responses.get(&key).cloned())
    }

    fn list(&self, tenant_id: &str) -> Result<Vec<StoredOpenAiResponse>, String> {
        let mut responses = if let Some(kv) = &self.kv {
            kv.list_json(RESPONSES_STORE_NAMESPACE, tenant_id)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(serde_json::from_value::<StoredOpenAiResponse>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?
        } else {
            self.memory
                .lock()
                .map_err(|_| "responses store lock poisoned".to_string())?
                .values()
                .filter(|response| response.tenant_id == tenant_id)
                .cloned()
                .collect::<Vec<_>>()
        };
        responses.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        Ok(responses)
    }

    async fn delete(&self, tenant_id: &str, response_id: &str) -> Result<bool, String> {
        let mut deleted = false;
        if let Some(kv) = &self.kv {
            deleted = kv
                .delete_json(RESPONSES_STORE_NAMESPACE, tenant_id, response_id)
                .await
                .map_err(|error| error.to_string())?;
        }

        let key = response_store_memory_key(tenant_id, response_id);
        let memory_deleted = self
            .memory
            .lock()
            .map_err(|_| "responses store lock poisoned".to_string())?
            .remove(&key)
            .is_some();
        Ok(deleted || memory_deleted)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredOpenAiResponse {
    tenant_id: String,
    id: String,
    created_at: u64,
    response: serde_json::Value,
    conversation_messages: Vec<OpenAiMessage>,
    input_items: Vec<serde_json::Value>,
}

fn response_store_memory_key(tenant_id: &str, response_id: &str) -> String {
    format!("{tenant_id}:{response_id}")
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
        let responses = ResponsesStore::new(context.store.clone());
        let metrics = RuntimeMetrics::default();
        Self {
            kernel: Arc::new(KernelRunner::new(provider, policy)),
            context,
            responses,
            direct,
            tenant_limiter,
            training_trace,
            public_reasoning_mode: PublicReasoningMode::Request,
            shared_api_key: None,
            durable: DurableStore::memory(),
            jobs: JobRuntime::new(4, metrics.clone()),
            metrics,
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
pub struct ProviderAdmission {
    semaphore: Option<Arc<Semaphore>>,
    wait_timeout: Duration,
}

impl ProviderAdmission {
    pub fn disabled() -> Self {
        Self {
            semaphore: None,
            wait_timeout: Duration::from_secs(30),
        }
    }

    fn from_env() -> Result<Self, String> {
        let max_concurrent = env_optional_usize(
            &[
                "MIYA_PROVIDER_MAX_CONCURRENT",
                "PROVIDER_MAX_CONCURRENT_REQUESTS",
            ],
            64,
        )?;
        let wait_timeout_ms = env_optional_u64(
            &[
                "MIYA_PROVIDER_QUEUE_TIMEOUT_MS",
                "PROVIDER_QUEUE_TIMEOUT_MS",
            ],
            30_000,
        )?;
        Ok(Self {
            semaphore: (max_concurrent > 0).then(|| Arc::new(Semaphore::new(max_concurrent))),
            wait_timeout: Duration::from_millis(wait_timeout_ms.max(1)),
        })
    }

    async fn acquire(&self) -> Result<ProviderAdmissionPermit, String> {
        let Some(semaphore) = &self.semaphore else {
            return Ok(ProviderAdmissionPermit { _permit: None });
        };
        let permit = tokio::time::timeout(self.wait_timeout, semaphore.clone().acquire_owned())
            .await
            .map_err(|_| {
                format!(
                    "provider queue wait exceeded {} ms",
                    self.wait_timeout.as_millis()
                )
            })?
            .map_err(|_| "provider concurrency limiter was closed".to_string())?;
        Ok(ProviderAdmissionPermit {
            _permit: Some(permit),
        })
    }
}

struct ProviderAdmissionPermit {
    _permit: Option<OwnedSemaphorePermit>,
}

#[derive(Clone)]
struct AdmissionProvider {
    inner: Arc<dyn ModelProvider>,
    admission: ProviderAdmission,
}

#[async_trait::async_trait]
impl ModelProvider for AdmissionProvider {
    async fn invoke(
        &self,
        request: provider_core::ProviderRequest,
    ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
        let _permit =
            self.admission
                .acquire()
                .await
                .map_err(|message| ProviderError::QueueTimeout {
                    wait_ms: provider_queue_wait_ms(&message),
                })?;
        self.inner.invoke(request).await
    }

    async fn stream(
        &self,
        request: provider_core::ProviderRequest,
    ) -> Result<ProviderStream, provider_core::ProviderError> {
        let permit =
            self.admission
                .acquire()
                .await
                .map_err(|message| ProviderError::QueueTimeout {
                    wait_ms: provider_queue_wait_ms(&message),
                })?;
        let stream = self.inner.stream(request).await?;
        Ok(Box::pin(stream.map(move |event| {
            let _hold_permit = &permit;
            event
        })))
    }
}

fn provider_queue_wait_ms(message: &str) -> u64 {
    message
        .split_whitespace()
        .find_map(|part| part.parse::<u64>().ok())
        .unwrap_or(30_000)
}

#[derive(Clone, Debug)]
pub struct ResilienceConfig {
    max_retries: usize,
    base_delay: Duration,
    circuit_failure_threshold: u32,
    circuit_cooldown: Duration,
}

impl ResilienceConfig {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            max_retries: env_optional_usize(&["MIYA_PROVIDER_MAX_RETRIES"], 2)?,
            base_delay: Duration::from_millis(env_optional_u64(
                &["MIYA_PROVIDER_RETRY_BASE_MS"],
                250,
            )?),
            circuit_failure_threshold: env_optional_u64(
                &["MIYA_PROVIDER_CIRCUIT_FAILURE_THRESHOLD"],
                5,
            )?
            .try_into()
            .map_err(|_| "MIYA_PROVIDER_CIRCUIT_FAILURE_THRESHOLD is too large".to_string())?,
            circuit_cooldown: Duration::from_millis(env_optional_u64(
                &["MIYA_PROVIDER_CIRCUIT_COOLDOWN_MS"],
                30_000,
            )?),
        })
    }
}

#[derive(Clone, Default)]
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
}

#[derive(Debug, Default)]
struct CircuitState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
    half_open_probe: bool,
}

impl CircuitBreaker {
    fn before_request(&self) -> Result<(), ProviderError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ProviderError::Rejected("circuit breaker lock poisoned".to_string()))?;
        let Some(open_until) = state.open_until else {
            return Ok(());
        };
        let now = Instant::now();
        if now < open_until {
            return Err(ProviderError::CircuitOpen {
                retry_after_ms: open_until.duration_since(now).as_millis() as u64,
            });
        }
        if state.half_open_probe {
            return Err(ProviderError::CircuitOpen {
                retry_after_ms: 100,
            });
        }
        state.half_open_probe = true;
        Ok(())
    }

    fn record_success(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = CircuitState::default();
        }
    }

    fn record_failure(&self, config: &ResilienceConfig) {
        if let Ok(mut state) = self.state.lock() {
            state.consecutive_failures = state.consecutive_failures.saturating_add(1);
            if state.half_open_probe
                || state.consecutive_failures >= config.circuit_failure_threshold.max(1)
            {
                state.open_until = Some(Instant::now() + config.circuit_cooldown);
                state.half_open_probe = false;
            }
        }
    }
}

#[derive(Clone)]
struct ResilientProvider {
    inner: Arc<dyn ModelProvider>,
    config: ResilienceConfig,
    circuit: CircuitBreaker,
    metrics: RuntimeMetrics,
}

#[async_trait::async_trait]
impl ModelProvider for ResilientProvider {
    async fn invoke(
        &self,
        request: provider_core::ProviderRequest,
    ) -> Result<provider_core::ProviderResponse, ProviderError> {
        self.circuit.before_request().inspect_err(|_| {
            self.metrics
                .circuit_rejections
                .fetch_add(1, Ordering::Relaxed);
        })?;
        let mut attempt = 0_usize;
        loop {
            self.metrics
                .provider_attempts
                .fetch_add(1, Ordering::Relaxed);
            match self.inner.invoke(request.clone()).await {
                Ok(response) => {
                    self.circuit.record_success();
                    return Ok(response);
                }
                Err(error) if error.retryable() && attempt < self.config.max_retries => {
                    self.metrics
                        .provider_retries
                        .fetch_add(1, Ordering::Relaxed);
                    let delay = provider_retry_delay(&self.config, &error, attempt);
                    attempt += 1;
                    tokio::time::sleep(delay).await;
                }
                Err(error) => {
                    self.metrics
                        .provider_failures
                        .fetch_add(1, Ordering::Relaxed);
                    if error.retryable() {
                        self.circuit.record_failure(&self.config);
                    } else {
                        self.circuit.record_success();
                    }
                    return Err(error);
                }
            }
        }
    }

    async fn stream(
        &self,
        request: provider_core::ProviderRequest,
    ) -> Result<ProviderStream, ProviderError> {
        self.circuit.before_request().inspect_err(|_| {
            self.metrics
                .circuit_rejections
                .fetch_add(1, Ordering::Relaxed);
        })?;
        let mut attempt = 0_usize;
        loop {
            self.metrics
                .provider_attempts
                .fetch_add(1, Ordering::Relaxed);
            match self.inner.stream(request.clone()).await {
                Ok(stream) => {
                    self.circuit.record_success();
                    return Ok(stream);
                }
                Err(error) if error.retryable() && attempt < self.config.max_retries => {
                    self.metrics
                        .provider_retries
                        .fetch_add(1, Ordering::Relaxed);
                    let delay = provider_retry_delay(&self.config, &error, attempt);
                    attempt += 1;
                    tokio::time::sleep(delay).await;
                }
                Err(error) => {
                    self.metrics
                        .provider_failures
                        .fetch_add(1, Ordering::Relaxed);
                    if error.retryable() {
                        self.circuit.record_failure(&self.config);
                    } else {
                        self.circuit.record_success();
                    }
                    return Err(error);
                }
            }
        }
    }
}

fn provider_retry_delay(
    config: &ResilienceConfig,
    error: &ProviderError,
    attempt: usize,
) -> Duration {
    if let Some(retry_after_ms) = error.retry_after_ms() {
        return Duration::from_millis(retry_after_ms.min(120_000));
    }
    let exponent = attempt.min(10) as u32;
    let base_ms = config.base_delay.as_millis() as u64;
    let jitter = telemetry_timestamp_ms() % 101;
    Duration::from_millis(
        base_ms
            .saturating_mul(2_u64.saturating_pow(exponent))
            .saturating_add(jitter)
            .min(30_000),
    )
}

#[derive(Clone)]
pub enum DirectBackend {
    Mock,
    OpenAi {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        admission: ProviderAdmission,
        resilience: ResilienceConfig,
        circuit: CircuitBreaker,
        metrics: RuntimeMetrics,
    },
    Anthropic {
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        api_version: String,
        admission: ProviderAdmission,
        resilience: ResilienceConfig,
        circuit: CircuitBreaker,
        metrics: RuntimeMetrics,
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
    wait_timeout: Duration,
    semaphores: Arc<Mutex<HashMap<String, Weak<Semaphore>>>>,
}

impl TenantConcurrencyLimiter {
    fn disabled() -> Self {
        Self {
            max_per_tenant: None,
            wait_timeout: Duration::from_millis(DEFAULT_TENANT_QUEUE_TIMEOUT_MS),
            semaphores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn with_max_per_tenant(max_per_tenant: usize) -> Self {
        Self::with_max_per_tenant_and_wait(
            max_per_tenant,
            Duration::from_millis(DEFAULT_TENANT_QUEUE_TIMEOUT_MS),
        )
    }

    fn with_max_per_tenant_and_wait(max_per_tenant: usize, wait_timeout: Duration) -> Self {
        if max_per_tenant == 0 {
            return Self::disabled();
        }
        Self {
            max_per_tenant: Some(max_per_tenant),
            wait_timeout,
            semaphores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn from_env() -> Self {
        let env_value = std::env::var("TENANT_MAX_CONCURRENT_REQUESTS").ok();
        let wait_timeout_ms = std::env::var("MIYA_TENANT_QUEUE_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_TENANT_QUEUE_TIMEOUT_MS);
        Self::from_env_values(env_value.as_deref(), wait_timeout_ms)
    }

    fn from_env_values(value: Option<&str>, wait_timeout_ms: u64) -> Self {
        let max = value
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS);
        Self::with_max_per_tenant_and_wait(max, Duration::from_millis(wait_timeout_ms.max(1)))
    }

    async fn acquire(
        &self,
        tenant_id: &TenantId,
    ) -> Result<TenantConcurrencyPermit, ProviderError> {
        let Some(max_per_tenant) = self.max_per_tenant else {
            return Ok(TenantConcurrencyPermit { _permit: None });
        };
        let semaphore = {
            let mut semaphores = self.semaphores.lock().map_err(|_| {
                ProviderError::Rejected("tenant concurrency limiter lock poisoned".to_string())
            })?;
            semaphores.retain(|_, semaphore| semaphore.strong_count() > 0);
            let tenant_key = tenant_id.as_ref().to_string();
            if let Some(semaphore) = semaphores.get(&tenant_key).and_then(Weak::upgrade) {
                semaphore
            } else {
                let semaphore = Arc::new(Semaphore::new(max_per_tenant));
                semaphores.insert(tenant_key, Arc::downgrade(&semaphore));
                semaphore
            }
        };
        let permit = tokio::time::timeout(self.wait_timeout, semaphore.acquire_owned())
            .await
            .map_err(|_| ProviderError::QueueTimeout {
                wait_ms: self.wait_timeout.as_millis().min(u128::from(u64::MAX)) as u64,
            })?
            .map_err(|_| {
                ProviderError::Rejected("tenant concurrency limiter was closed".to_string())
            })?;
        Ok(TenantConcurrencyPermit {
            _permit: Some(permit),
        })
    }
}

struct TenantConcurrencyPermit {
    _permit: Option<OwnedSemaphorePermit>,
}

fn direct_http_client() -> Result<reqwest::Client, String> {
    let request_timeout_secs = env_optional_u64(&["MIYA_PROVIDER_TIMEOUT_SECS"], 300)?;
    let connect_timeout_secs = env_optional_u64(&["MIYA_PROVIDER_CONNECT_TIMEOUT_SECS"], 30)?;
    reqwest::Client::builder()
        .timeout(Duration::from_secs(request_timeout_secs.max(1)))
        .connect_timeout(Duration::from_secs(connect_timeout_secs.max(1)))
        .build()
        .map_err(|error| format!("failed to build direct provider HTTP client: {error}"))
}

impl DirectBackend {
    fn from_env(
        provider_kind: &str,
        admission: ProviderAdmission,
        resilience: ResilienceConfig,
        metrics: RuntimeMetrics,
    ) -> Result<Self, String> {
        match provider_kind {
            "mock" => Ok(Self::Mock),
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY is required".to_string())?;
                let base_url = std::env::var("OPENAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
                Ok(Self::OpenAi {
                    client: direct_http_client()?,
                    base_url: base_url.trim_end_matches('/').to_string(),
                    api_key,
                    admission,
                    resilience,
                    circuit: CircuitBreaker::default(),
                    metrics,
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
                    client: direct_http_client()?,
                    base_url: base_url.trim_end_matches('/').to_string(),
                    api_key,
                    api_version,
                    admission,
                    resilience,
                    circuit: CircuitBreaker::default(),
                    metrics,
                })
            }
            other => Err(format!("unsupported direct provider={other}")),
        }
    }

    async fn openai_chat(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, ProviderError> {
        match self {
            Self::Mock => Ok(mock_direct_openai_response(&request)),
            Self::OpenAi {
                client,
                base_url,
                api_key,
                admission,
                resilience,
                circuit,
                metrics,
            } => {
                let request = sanitize_direct_openai_request(request);
                let tool_names = openai_request_tool_names(&request);
                resilient_direct_json(
                    "openai",
                    admission,
                    resilience,
                    circuit,
                    metrics,
                    || {
                        client
                            .post(format!("{base_url}/chat/completions"))
                            .bearer_auth(api_key)
                            .json(&request)
                    },
                )
                .await
                .map(|response| strip_direct_openai_response_with_tools(response, &tool_names))
            }
            Self::Anthropic { .. } => Err(ProviderError::Rejected(
                "reasoning.effort=none on /v1/chat/completions requires MULTI_AGENT_PROVIDER=openai"
                    .to_string(),
            )),
        }
    }

    async fn openai_chat_stream(
        &self,
        request: serde_json::Value,
    ) -> Result<Response, ProviderError> {
        match self {
            Self::Mock => Ok(openai_stream_response_from_completion(
                mock_direct_openai_response(&request),
            )),
            Self::OpenAi {
                client,
                base_url,
                api_key,
                admission,
                resilience,
                circuit,
                metrics,
            } => {
                let request = sanitize_direct_openai_request(request);
                let (response, provider_permit) = resilient_direct_send(
                    "openai",
                    admission,
                    resilience,
                    circuit,
                    metrics,
                    || {
                        client
                            .post(format!("{base_url}/chat/completions"))
                            .bearer_auth(api_key)
                            .json(&request)
                    },
                )
                .await?;
                Ok(response_with_provider_permit(
                    upstream_sse_response(response),
                    provider_permit,
                ))
            }
            Self::Anthropic { .. } => Err(ProviderError::Rejected(
                "reasoning.effort=none on /v1/chat/completions requires MULTI_AGENT_PROVIDER=openai"
                    .to_string(),
            )),
        }
    }

    async fn anthropic_messages(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, ProviderError> {
        match self {
            Self::Mock => Ok(mock_direct_anthropic_response(&request)),
            Self::Anthropic {
                client,
                base_url,
                api_key,
                api_version,
                admission,
                resilience,
                circuit,
                metrics,
            } => {
                let request = sanitize_direct_anthropic_request(request);
                resilient_direct_json("anthropic", admission, resilience, circuit, metrics, || {
                    client
                        .post(format!("{base_url}/v1/messages"))
                        .header("x-api-key", api_key)
                        .header("anthropic-version", api_version)
                        .json(&request)
                })
                .await
                .map(strip_direct_anthropic_response)
            }
            Self::OpenAi { .. } => Err(ProviderError::Rejected(
                "reasoning.effort=none on /v1/messages requires MULTI_AGENT_PROVIDER=anthropic"
                    .to_string(),
            )),
        }
    }

    async fn anthropic_messages_stream(
        &self,
        request: serde_json::Value,
    ) -> Result<Response, ProviderError> {
        match self {
            Self::Mock => Ok(anthropic_stream_response_from_message(
                mock_direct_anthropic_response(&request),
            )),
            Self::Anthropic {
                client,
                base_url,
                api_key,
                api_version,
                admission,
                resilience,
                circuit,
                metrics,
            } => {
                let request = sanitize_direct_anthropic_request(request);
                let (response, provider_permit) = resilient_direct_send(
                    "anthropic",
                    admission,
                    resilience,
                    circuit,
                    metrics,
                    || {
                        client
                            .post(format!("{base_url}/v1/messages"))
                            .header("x-api-key", api_key)
                            .header("anthropic-version", api_version)
                            .json(&request)
                    },
                )
                .await?;
                Ok(response_with_provider_permit(
                    upstream_sse_response(response),
                    provider_permit,
                ))
            }
            Self::OpenAi { .. } => Err(ProviderError::Rejected(
                "reasoning.effort=none on /v1/messages requires MULTI_AGENT_PROVIDER=anthropic"
                    .to_string(),
            )),
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

async fn resilient_direct_json<F>(
    provider: &str,
    admission: &ProviderAdmission,
    resilience: &ResilienceConfig,
    circuit: &CircuitBreaker,
    metrics: &RuntimeMetrics,
    build: F,
) -> Result<serde_json::Value, ProviderError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    let (response, _permit) =
        resilient_direct_send(provider, admission, resilience, circuit, metrics, build).await?;
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| ProviderError::InvalidResponse {
            provider: provider.to_string(),
            message: error.to_string(),
        })
}

async fn resilient_direct_send<F>(
    provider: &str,
    admission: &ProviderAdmission,
    resilience: &ResilienceConfig,
    circuit: &CircuitBreaker,
    metrics: &RuntimeMetrics,
    build: F,
) -> Result<(reqwest::Response, ProviderAdmissionPermit), ProviderError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    circuit.before_request().inspect_err(|_| {
        metrics.circuit_rejections.fetch_add(1, Ordering::Relaxed);
    })?;
    let mut attempt = 0_usize;
    loop {
        metrics.provider_attempts.fetch_add(1, Ordering::Relaxed);
        let result = async {
            let permit =
                admission
                    .acquire()
                    .await
                    .map_err(|message| ProviderError::QueueTimeout {
                        wait_ms: provider_queue_wait_ms(&message),
                    })?;
            let response = build()
                .send()
                .await
                .map_err(|error| ProviderError::Transport {
                    provider: provider.to_string(),
                    message: error.to_string(),
                    retryable: error.is_timeout() || error.is_connect() || error.is_request(),
                })?;
            let response = direct_response_or_error(response, provider).await?;
            Ok::<_, ProviderError>((response, permit))
        }
        .await;
        match result {
            Ok(result) => {
                circuit.record_success();
                return Ok(result);
            }
            Err(error) if error.retryable() && attempt < resilience.max_retries => {
                metrics.provider_retries.fetch_add(1, Ordering::Relaxed);
                let delay = provider_retry_delay(resilience, &error, attempt);
                attempt += 1;
                tokio::time::sleep(delay).await;
            }
            Err(error) => {
                metrics.provider_failures.fetch_add(1, Ordering::Relaxed);
                if error.retryable() {
                    circuit.record_failure(resilience);
                } else {
                    circuit.record_success();
                }
                return Err(error);
            }
        }
    }
}

async fn direct_response_or_error(
    response: reqwest::Response,
    provider: &str,
) -> Result<reqwest::Response, ProviderError> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status().as_u16();
    let retry_after_ms = response
        .headers()
        .get(header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|seconds| seconds.saturating_mul(1000));
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
    let value = serde_json::from_str::<serde_json::Value>(&body).ok();
    let error = value.as_ref().and_then(|value| value.get("error"));
    let code = error
        .and_then(|error| error.get("code").or_else(|| error.get("type")))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or(body);
    Err(ProviderError::Http {
        provider: provider.to_string(),
        status,
        code,
        message,
        retry_after_ms,
    })
}

fn sanitize_direct_openai_request(mut request: serde_json::Value) -> serde_json::Value {
    strip_gateway_metadata(&mut request);
    strip_gateway_reasoning_effort(&mut request);
    strip_gateway_public_reasoning_options(&mut request);
    normalize_direct_openai_tools(&mut request);
    adapt_direct_named_tool_choice(&mut request);
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
    if let Some(object) = request.as_object_mut() {
        object.remove("reasoning_effort");
    }

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

fn adapt_direct_named_tool_choice(request: &mut serde_json::Value) {
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

#[cfg(test)]
fn strip_direct_openai_response(response: serde_json::Value) -> serde_json::Value {
    strip_direct_openai_response_with_tools(response, &BTreeSet::new())
}

fn strip_direct_openai_response_with_tools(
    mut response: serde_json::Value,
    tool_names: &BTreeSet<String>,
) -> serde_json::Value {
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
                if let Some(text) = message
                    .get("content")
                    .and_then(|content| content.as_str())
                    .map(strip_direct_thinking_markup)
                {
                    if !message.contains_key("tool_calls")
                        && let Some(call) = parse_text_tool_call(&text, tool_names)
                    {
                        message.insert(
                            "tool_calls".to_string(),
                            serde_json::json!([openai_tool_call_json(
                                parsed_text_tool_call_record(call)
                            )]),
                        );
                        message.insert(
                            "content".to_string(),
                            serde_json::Value::String(String::new()),
                        );
                        if let Some(object) = choice.as_object_mut() {
                            object.insert(
                                "finish_reason".to_string(),
                                serde_json::Value::String("tool_calls".to_string()),
                            );
                        }
                    } else {
                        message.insert("content".to_string(), serde_json::Value::String(text));
                    }
                }
            }
        }
    }
    response
}

fn openai_request_tool_names(request: &serde_json::Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if let Some(tools) = request.get("tools").and_then(|value| value.as_array()) {
        for tool in tools {
            if let Some(name) = tool
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(|value| value.as_str())
            {
                names.insert(name.to_string());
            }
        }
    }
    if let Some(functions) = request.get("functions").and_then(|value| value.as_array()) {
        for function in functions {
            if let Some(name) = function.get("name").and_then(|value| value.as_str()) {
                names.insert(name.to_string());
            }
        }
    }
    names
}

fn normalized_tool_names(tools: &[ToolDefinition]) -> BTreeSet<String> {
    tools.iter().map(|tool| tool.name.clone()).collect()
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

#[derive(Clone, Debug, PartialEq)]
struct ParsedTextToolCall {
    name: String,
    arguments: serde_json::Value,
}

fn parse_text_tool_call(
    text: &str,
    available_tool_names: &BTreeSet<String>,
) -> Option<ParsedTextToolCall> {
    if available_tool_names.is_empty() {
        return None;
    }
    let text = text.trim();
    let inner = extract_text_tool_call_inner(text)?;
    let call = parse_text_tool_call_inner(inner)?;
    if available_tool_names.contains(&call.name) {
        return Some(call);
    }
    adapt_unknown_text_tool_call_to_available_tool(call, available_tool_names)
}

fn adapt_unknown_text_tool_call_to_available_tool(
    call: ParsedTextToolCall,
    available_tool_names: &BTreeSet<String>,
) -> Option<ParsedTextToolCall> {
    let exec_name = ["exec_command", "local_shell", "shell"]
        .iter()
        .find(|name| available_tool_names.contains(**name))?;
    let message = format!(
        "Unsupported model-emitted tool '{}' was ignored. Continue using only declared tools.",
        call.name
    );
    Some(ParsedTextToolCall {
        name: (*exec_name).to_string(),
        arguments: serde_json::json!({
            "cmd": format!("printf '%s\\n' {}", shell_quote(&message))
        }),
    })
}

fn extract_text_tool_call_inner(text: &str) -> Option<&str> {
    [
        ("<|tool_call>", "<tool_call|>"),
        ("<tool_call>", "</tool_call>"),
        ("<tool_call>", "<tool_call|>"),
    ]
    .iter()
    .find_map(|(start_marker, end_marker)| {
        let start = text.find(start_marker)? + start_marker.len();
        let end = text[start..].find(end_marker)? + start;
        Some(text[start..end].trim())
    })
}

fn parse_text_tool_call_inner(inner: &str) -> Option<ParsedTextToolCall> {
    let inner = inner.trim();
    let normalized = normalize_text_tool_call_quotes(inner);
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&normalized)
        && let Some(call) = parse_text_tool_call_json(value)
    {
        return Some(call);
    }

    let body = inner
        .strip_prefix("call:")
        .or_else(|| inner.strip_prefix("function:"))
        .unwrap_or(inner);
    let open = body.find('{')?;
    let close = body.rfind('}')?;
    if close <= open {
        return None;
    }
    let name = sanitize_tool_name(body[..open].trim());
    if name.is_empty() {
        return None;
    }
    let arguments = parse_text_tool_call_arguments(&body[open..=close])?;
    Some(ParsedTextToolCall { name, arguments })
}

fn parse_text_tool_call_json(value: serde_json::Value) -> Option<ParsedTextToolCall> {
    let object = value.as_object()?;
    let name = object
        .get("name")
        .or_else(|| object.get("tool_name"))
        .or_else(|| {
            object
                .get("function")
                .and_then(|function| function.get("name"))
        })
        .and_then(|value| value.as_str())
        .map(sanitize_tool_name)?;
    if name.is_empty() {
        return None;
    }
    let arguments = object
        .get("arguments")
        .or_else(|| object.get("args"))
        .or_else(|| object.get("parameters"))
        .or_else(|| {
            object
                .get("function")
                .and_then(|function| function.get("arguments"))
        })
        .cloned()
        .map(openai_tool_arguments)
        .unwrap_or_else(|| serde_json::json!({}));
    Some(ParsedTextToolCall { name, arguments })
}

fn normalize_text_tool_call_quotes(text: &str) -> String {
    text.replace("<|\"|>", "\"")
        .replace("<|quote|>", "\"")
        .replace("&quot;", "\"")
}

fn parse_text_tool_call_arguments(text: &str) -> Option<serde_json::Value> {
    if let Ok(value) =
        serde_json::from_str::<serde_json::Value>(&normalize_text_tool_call_quotes(text))
    {
        return Some(openai_tool_arguments(value));
    }
    let trimmed = text.trim();
    let inner = trimmed.strip_prefix('{')?.strip_suffix('}')?;
    let chars = inner.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut object = serde_json::Map::new();

    while index < chars.len() {
        index = skip_tool_call_separators(&chars, index);
        if index >= chars.len() {
            break;
        }

        let (key, after_key) = parse_tool_call_key(&chars, index)?;
        index = skip_summary_whitespace(&chars, after_key);
        if !matches!(chars.get(index), Some(':') | Some('=')) {
            return None;
        }
        index = skip_summary_whitespace(&chars, index + 1);
        let (value, after_value) = parse_tool_call_value(&chars, index)?;
        object.insert(key, value);
        index = after_value;
    }

    (!object.is_empty()).then_some(serde_json::Value::Object(object))
}

fn skip_tool_call_separators(chars: &[char], mut index: usize) -> usize {
    while chars
        .get(index)
        .is_some_and(|ch| ch.is_whitespace() || *ch == ',')
    {
        index += 1;
    }
    index
}

fn parse_tool_call_key(chars: &[char], index: usize) -> Option<(String, usize)> {
    if chars.get(index) == Some(&'"') {
        return parse_quoted_fragment(chars, index);
    }
    let mut end = index;
    while end < chars.len() && !matches!(chars[end], ':' | '=' | ',' | '}') {
        end += 1;
    }
    let key = chars[index..end]
        .iter()
        .collect::<String>()
        .trim()
        .to_string();
    (!key.is_empty()).then_some((key, end))
}

fn parse_tool_call_value(chars: &[char], index: usize) -> Option<(serde_json::Value, usize)> {
    if starts_with_chars(chars, index, "<|\"|>") {
        let (value, after) = parse_marker_quoted_fragment(chars, index, "<|\"|>")?;
        return Some((serde_json::Value::String(value), after));
    }
    if starts_with_chars(chars, index, "<|quote|>") {
        let (value, after) = parse_marker_quoted_fragment(chars, index, "<|quote|>")?;
        return Some((serde_json::Value::String(value), after));
    }
    if chars.get(index) == Some(&'"') {
        let (value, after) = parse_quoted_fragment(chars, index)?;
        return Some((serde_json::Value::String(value), after));
    }

    let mut end = index;
    while end < chars.len() && chars[end] != ',' {
        end += 1;
    }
    let raw = chars[index..end]
        .iter()
        .collect::<String>()
        .trim()
        .to_string();
    if raw.is_empty() {
        return None;
    }
    let value =
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or(serde_json::Value::String(raw));
    Some((value, end))
}

fn starts_with_chars(chars: &[char], index: usize, marker: &str) -> bool {
    marker
        .chars()
        .enumerate()
        .all(|(offset, ch)| chars.get(index + offset) == Some(&ch))
}

fn parse_marker_quoted_fragment(
    chars: &[char],
    start: usize,
    marker: &str,
) -> Option<(String, usize)> {
    if !starts_with_chars(chars, start, marker) {
        return None;
    }
    let marker_len = marker.chars().count();
    let mut index = start + marker_len;
    let mut output = String::new();
    while index < chars.len() {
        if starts_with_chars(chars, index, marker) {
            return Some((output, index + marker_len));
        }
        output.push(chars[index]);
        index += 1;
    }
    None
}

fn parsed_text_tool_call_record(call: ParsedTextToolCall) -> ToolCallRecord {
    let arguments_sha256 = sha256_hex(&call.arguments.to_string());
    ToolCallRecord {
        tool_call_id: ToolCallId::from(format!("call_{}", Uuid::new_v4().simple())),
        scope: IsolationKey::new("default", "text-tool-call", "root"),
        task_id: TaskId::from("root"),
        agent_id: AgentId::from("agent-root"),
        tool_name: call.name,
        arguments_sha256,
        arguments_json: call.arguments,
        status: ToolCallStatus::Pending,
        created_at_ms: unix_timestamp_secs().saturating_mul(1000),
        resolved_at_ms: None,
    }
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

pub fn init_observability() -> Result<ObservabilityGuard, String> {
    observability::init()
}

pub fn build_router() -> Router {
    build_router_with_state(AppState::default())
}

pub fn build_router_from_env() -> Result<Router, String> {
    let provider_kind =
        std::env::var("MULTI_AGENT_PROVIDER").unwrap_or_else(|_| "mock".to_string());
    let provider = provider_from_kind(&provider_kind)?;
    let provider_admission = ProviderAdmission::from_env()?;
    let metrics = RuntimeMetrics::default();
    let resilience = ResilienceConfig::from_env()?;
    let admitted_provider: Arc<dyn ModelProvider> = Arc::new(AdmissionProvider {
        inner: provider,
        admission: provider_admission.clone(),
    });
    let provider: Arc<dyn ModelProvider> = Arc::new(ResilientProvider {
        inner: admitted_provider,
        config: resilience.clone(),
        circuit: CircuitBreaker::default(),
        metrics: metrics.clone(),
    });
    let context = ApiContextManager::from_env()?;
    let direct = DirectBackend::from_env(
        &provider_kind,
        provider_admission,
        resilience,
        metrics.clone(),
    )?;
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
    state.shared_api_key = shared_api_key_from_env()?;
    state.metrics = metrics;
    let data_dir = std::env::var("MIYA_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".multi-agent-data"));
    state.durable = DurableStore::filesystem(data_dir.clone())?;
    if state.context.store.is_none() {
        let response_store = SurrealKvContextStore::open(data_dir.join("responses"))
            .map_err(|error| error.to_string())?;
        state.responses = ResponsesStore::new(Some(Arc::new(response_store)));
    }
    let max_jobs = env_optional_usize(&["MIYA_MAX_CONCURRENT_JOBS"], 4)?;
    state.jobs = JobRuntime::new(max_jobs, state.metrics.clone());
    Ok(build_router_with_state(state))
}

fn shared_api_key_from_env() -> Result<Option<Arc<str>>, String> {
    for name in ["MIYA_API_KEY", "MIYA_SHARED_API_KEY"] {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            return Err(format!("{name} must not be empty when configured"));
        }
        return Ok(Some(Arc::<str>::from(value)));
    }
    Ok(None)
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
    if let Some(request_timeout_ms) =
        env_u64(&["MIYA_REQUEST_TIMEOUT_MS", "MULTI_AGENT_REQUEST_TIMEOUT_MS"])?
    {
        policy.limits.request_timeout_ms = request_timeout_ms;
    }
    if let Some(agent_timeout_ms) =
        env_u64(&["MIYA_AGENT_TIMEOUT_MS", "MULTI_AGENT_AGENT_TIMEOUT_MS"])?
    {
        policy.limits.agent_timeout_ms = agent_timeout_ms;
    }
    policy.semantic_verification.enabled = std::env::var("MIYA_SEMANTIC_VERIFIER")
        .map(|value| env_flag_enabled(&value))
        .unwrap_or(true);
    policy.semantic_verification.max_repair_attempts = env_optional_u64(
        &["MIYA_SEMANTIC_MAX_REPAIR_ATTEMPTS"],
        u64::from(policy.semantic_verification.max_repair_attempts),
    )?
    .try_into()
    .map_err(|_| "MIYA_SEMANTIC_MAX_REPAIR_ATTEMPTS must fit in u8".to_string())?;
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

fn env_u64(names: &[&str]) -> Result<Option<u64>, String> {
    for name in names {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        let parsed = value
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("{name} must be a positive integer"))?;
        if parsed == 0 {
            return Err(format!("{name} must be greater than 0"));
        }
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn env_optional_u64(names: &[&str], default: u64) -> Result<u64, String> {
    for name in names {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        return value
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("{name} must be a non-negative integer"));
    }
    Ok(default)
}

fn env_optional_usize(names: &[&str], default: usize) -> Result<usize, String> {
    let value = env_optional_u64(names, default as u64)?;
    usize::try_from(value).map_err(|_| format!("{} is too large", names[0]))
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
    let shared_key_enabled = state.shared_api_key.is_some();
    recover_durable_jobs(state.clone());
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/v1/health", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/v1/v1/models", get(models))
        .route("/models/{model_id}", get(model_retrieve))
        .route("/v1/models/{model_id}", get(model_retrieve))
        .route("/v1/v1/models/{model_id}", get(model_retrieve))
        .route("/metrics", get(metrics))
        .route("/v1/metrics", get(metrics))
        .route(
            "/files",
            get(files_list)
                .post(files_create)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_FILE_BYTES + 1024 * 1024)),
        )
        .route(
            "/v1/files",
            get(files_list)
                .post(files_create)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_FILE_BYTES + 1024 * 1024)),
        )
        .route(
            "/v1/v1/files",
            get(files_list)
                .post(files_create)
                .layer(DefaultBodyLimit::max(MAX_OPENAI_FILE_BYTES + 1024 * 1024)),
        )
        .route("/files/{file_id}/content", get(files_content))
        .route("/v1/files/{file_id}/content", get(files_content))
        .route("/v1/v1/files/{file_id}/content", get(files_content))
        .route("/files/{file_id}", get(files_retrieve).delete(files_delete))
        .route(
            "/v1/files/{file_id}",
            get(files_retrieve).delete(files_delete),
        )
        .route(
            "/v1/v1/files/{file_id}",
            get(files_retrieve).delete(files_delete),
        )
        .route(
            "/batches",
            get(openai_batches_list).post(openai_batches_create),
        )
        .route(
            "/v1/batches",
            get(openai_batches_list).post(openai_batches_create),
        )
        .route(
            "/v1/v1/batches",
            get(openai_batches_list).post(openai_batches_create),
        )
        .route("/batches/{batch_id}", get(openai_batches_retrieve))
        .route("/v1/batches/{batch_id}", get(openai_batches_retrieve))
        .route("/v1/v1/batches/{batch_id}", get(openai_batches_retrieve))
        .route("/batches/{batch_id}/cancel", post(openai_batches_cancel))
        .route("/v1/batches/{batch_id}/cancel", post(openai_batches_cancel))
        .route(
            "/v1/v1/batches/{batch_id}/cancel",
            post(openai_batches_cancel),
        )
        .route("/completions", post(completions))
        .route("/v1/completions", post(completions))
        .route("/v1/v1/completions", post(completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/v1/chat/completions", post(chat_completions))
        .route("/responses", post(responses).get(responses_get))
        .route("/v1/responses", post(responses).get(responses_get))
        .route("/v1/v1/responses", post(responses).get(responses_get))
        .route("/responses/input_tokens", post(responses_input_tokens))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .route(
            "/v1/v1/responses/input_tokens",
            post(responses_input_tokens),
        )
        .route("/responses/compact", post(responses_compact))
        .route("/v1/responses/compact", post(responses_compact))
        .route("/v1/v1/responses/compact", post(responses_compact))
        .route(
            "/responses/{response_id}/input_items",
            get(responses_input_items),
        )
        .route(
            "/v1/responses/{response_id}/input_items",
            get(responses_input_items),
        )
        .route(
            "/v1/v1/responses/{response_id}/input_items",
            get(responses_input_items),
        )
        .route("/responses/{response_id}/cancel", post(responses_cancel))
        .route("/v1/responses/{response_id}/cancel", post(responses_cancel))
        .route(
            "/v1/v1/responses/{response_id}/cancel",
            post(responses_cancel),
        )
        .route(
            "/responses/{response_id}",
            get(responses_retrieve).delete(responses_delete),
        )
        .route(
            "/v1/responses/{response_id}",
            get(responses_retrieve).delete(responses_delete),
        )
        .route(
            "/v1/v1/responses/{response_id}",
            get(responses_retrieve).delete(responses_delete),
        )
        .route("/messages", post(messages))
        .route("/v1/messages", post(messages))
        .route("/v1/v1/messages", post(messages))
        .route("/messages/count_tokens", post(messages_count_tokens))
        .route("/v1/messages/count_tokens", post(messages_count_tokens))
        .route("/v1/v1/messages/count_tokens", post(messages_count_tokens))
        .route(
            "/v1/messages/batches",
            get(anthropic_batches_list).post(anthropic_batches_create),
        )
        .route(
            "/v1/messages/batches/{batch_id}/results",
            get(anthropic_batches_results),
        )
        .route(
            "/v1/messages/batches/{batch_id}/cancel",
            post(anthropic_batches_cancel),
        )
        .route(
            "/v1/messages/batches/{batch_id}",
            get(anthropic_batches_retrieve).delete(anthropic_batches_delete),
        )
        .with_state(state.clone())
        .layer(middleware::from_fn(observability::trace_request))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            runtime_metrics_middleware,
        ));

    if shared_key_enabled {
        router.layer(middleware::from_fn_with_state(state, shared_api_key_auth))
    } else {
        router
    }
}

async fn runtime_metrics_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    state.metrics.http_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let response = next.run(request).await;
    state.metrics.http_latency_micros.fetch_add(
        started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    if response.status().is_client_error() || response.status().is_server_error() {
        state.metrics.http_failures.fetch_add(1, Ordering::Relaxed);
    }
    response
}

async fn shared_api_key_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if is_public_health_path(request.uri().path()) {
        return next.run(request).await;
    }

    let Some(expected) = state.shared_api_key.as_deref() else {
        return next.run(request).await;
    };
    let provided = shared_api_key_from_headers(request.headers());
    if provided.is_some_and(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes())) {
        return next.run(request).await;
    }

    shared_api_key_error_response(request.uri().path())
}

fn is_public_health_path(path: &str) -> bool {
    matches!(path, "/health" | "/v1/health" | "/v1/v1/health")
}

fn shared_api_key_from_headers(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let (scheme, credentials) = value.trim().split_once(' ')?;
            if scheme.eq_ignore_ascii_case("bearer") && !credentials.trim().is_empty() {
                Some(credentials.trim())
            } else {
                None
            }
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn shared_api_key_error_response(path: &str) -> Response {
    let response = if path.contains("/messages") {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid or missing shared API key"
                }
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(CompatibilityError {
                error: CompatibilityErrorBody {
                    message: "invalid or missing shared API key".to_string(),
                    r#type: "authentication_error".to_string(),
                    code: "invalid_api_key".to_string(),
                },
            }),
        )
            .into_response()
    };

    let mut response = response;
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        header::HeaderValue::from_static("Bearer"),
    );
    response
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
        .map(model_compatibility_json)
        .collect::<Vec<_>>();
    let first_id = data
        .first()
        .and_then(|model| model.get("id"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let last_id = data
        .last()
        .and_then(|model| model.get("id"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    Json(serde_json::json!({
        "object": "list",
        "data": data,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id
    }))
}

async fn model_retrieve(Path(model_id): Path<String>) -> Response {
    if configured_model_ids()
        .iter()
        .any(|configured| configured == &model_id)
    {
        Json(model_compatibility_json(model_id)).into_response()
    } else {
        not_found_response(format!("model not found: {model_id}"))
    }
}

fn model_compatibility_json(id: String) -> serde_json::Value {
    serde_json::json!({
        "id": id.clone(),
        "object": "model",
        "created": 0,
        "owned_by": "miya-api",
        "type": "model",
        "display_name": id,
        "created_at": "1970-01-01T00:00:00Z"
    })
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
        Err(error) => return provider_error_response(error),
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
            Err(error) => kernel_error_response(error),
        };
    }

    let metrics = state.metrics.clone();
    let response_future = async move {
        let _tenant_permit = tenant_permit;
        match state.kernel.run(normalized).await {
            Ok(output) => {
                let context_report =
                    match finalize_context_report(&state.context, &prepared_context, &output).await
                    {
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
            Err(error) => kernel_error_response(error),
        }
    };
    if stream {
        orchestration_sse_response(response_future, metrics)
    } else {
        response_future.await
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
        Err(error) => return provider_error_response(error),
    };
    let direct_training_request = state
        .training_trace
        .capture_openai_request(&request, &request_context);
    if matches!(openai_reasoning_effort(&request), Ok(ReasoningEffort::None)) {
        let provider_raw_request = raw_request_with_provider_model(raw_request, &request.model);
        if stream {
            return match state.direct.openai_chat_stream(provider_raw_request).await {
                Ok(response) => response_with_tenant_permit(response, tenant_permit),
                Err(error) => provider_error_response(error),
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
            Err(error) => provider_error_response(error),
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
    let response_tool_names = normalized_tool_names(&normalized.tools);

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
            Err(error) => kernel_error_response(error),
        };
    }

    let metrics = state.metrics.clone();
    let response_future = async move {
        let _tenant_permit = tenant_permit;
        match state.kernel.run(normalized).await {
            Ok(output) => {
                let context_report =
                    match finalize_context_report(&state.context, &prepared_context, &output).await
                    {
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
                        &response_tool_names,
                        tool_response_format,
                        include_public_reasoning,
                    );
                }
                let mut response = format_openai_response(
                    model,
                    output,
                    &response_tool_names,
                    include_encrypted_state,
                    tool_response_format,
                    include_public_reasoning,
                );
                attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                Json(response).into_response()
            }
            Err(error) => kernel_error_response(error),
        }
    };
    if stream {
        orchestration_sse_response(response_future, metrics)
    } else {
        response_future.await
    }
}

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let persisted_request = raw_request.clone();
    let request = match parse_openai_responses_request(raw_request) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = request.stream;
    let include_public_reasoning = state
        .public_reasoning_mode
        .resolve(openai_responses_public_reasoning_requested(&request));
    let request_context =
        RequestContext::from_headers(&headers).with_metadata_overrides(&request.metadata);
    if request.background {
        return create_background_response(state, request_context, request, persisted_request)
            .await;
    }
    let tenant_id = request_context.tenant_id();
    let tenant_permit = match state.tenant_limiter.acquire(&tenant_id).await {
        Ok(permit) => permit,
        Err(error) => return provider_error_response(error),
    };
    let execution =
        match prepare_openai_responses_execution(&state.responses, request, &request_context).await
        {
            Ok(execution) => execution,
            Err(error) => return api_error_response(error),
        };

    if matches!(
        openai_reasoning_effort(&execution.chat_request),
        Ok(ReasoningEffort::None)
    ) {
        let raw_chat_request = openai_chat_request_json(&execution.chat_request, false);
        return match state.direct.openai_chat(raw_chat_request).await {
            Ok(value) => {
                let response =
                    openai_responses_response_from_chat_completion(&execution, value.clone());
                if let Err(error) = maybe_store_openai_response(
                    &state.responses,
                    &execution,
                    &response,
                    response_conversation_messages_from_chat_completion(
                        &execution,
                        execution.conversation_messages.clone(),
                        &value,
                    ),
                )
                .await
                {
                    return internal_error_response(error);
                }
                let telemetry_context = direct_telemetry_context(
                    "openai.responses",
                    execution.request_model.clone(),
                    "openai_responses",
                    &request_context,
                    stream,
                    None,
                );
                emit_direct_telemetry(&telemetry_context, response_usage(&value));
                let response = if stream {
                    format_openai_responses_stream_response_from_value(response)
                } else {
                    Json(response).into_response()
                };
                response_with_tenant_permit(response, tenant_permit)
            }
            Err(error) => provider_error_response(error),
        };
    }

    let model = execution.request_model.clone();
    let mut normalized = match normalize_openai_chat_with_context(
        execution.chat_request.clone(),
        &request_context,
    ) {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let telemetry_context =
        telemetry_context_from_normalized("openai.responses", &normalized, stream, None);
    let training_request = state.training_trace.capture_request(&normalized);

    if stream && !requires_full_orchestration_before_stream(&normalized, include_public_reasoning) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => response_with_tenant_permit(
                format_openai_responses_provider_stream_response(
                    execution,
                    provider_stream,
                    state.responses.clone(),
                    telemetry_context,
                ),
                tenant_permit,
            ),
            Err(error) => kernel_error_response(error),
        };
    }

    let metrics = state.metrics.clone();
    let response_future = async move {
        match state.kernel.run(normalized).await {
            Ok(output) => {
                let context_report =
                    match finalize_context_report(&state.context, &prepared_context, &output).await
                    {
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
                let conversation_messages = response_conversation_messages_from_kernel_output(
                    &execution,
                    execution.conversation_messages.clone(),
                    &output,
                );
                let mut response = format_openai_responses_response(
                    &execution,
                    &model,
                    output,
                    include_public_reasoning,
                );
                attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                if let Err(error) = maybe_store_openai_response(
                    &state.responses,
                    &execution,
                    &response,
                    conversation_messages,
                )
                .await
                {
                    return internal_error_response(error);
                }
                if stream {
                    return response_with_tenant_permit(
                        format_openai_responses_stream_response_from_value(response),
                        tenant_permit,
                    );
                }
                response_with_tenant_permit(Json(response).into_response(), tenant_permit)
            }
            Err(error) => kernel_error_response(error),
        }
    };
    if stream {
        orchestration_sse_response(response_future, metrics)
    } else {
        response_future.await
    }
}

async fn create_background_response(
    state: AppState,
    request_context: RequestContext,
    request: OpenAiResponsesRequest,
    mut persisted_request: serde_json::Value,
) -> Response {
    let execution =
        match prepare_openai_responses_execution(&state.responses, request, &request_context).await
        {
            Ok(execution) => execution,
            Err(error) => return api_error_response(error),
        };
    if let Some(object) = persisted_request.as_object_mut() {
        object.insert("background".to_string(), serde_json::Value::Bool(false));
        object.insert("stream".to_string(), serde_json::Value::Bool(false));
        object.insert("store".to_string(), serde_json::Value::Bool(true));
    }
    let persisted_bytes = match serde_json::to_vec(&persisted_request) {
        Ok(bytes) => bytes,
        Err(error) => return internal_error_response(error.to_string()),
    };
    let tenant_id = execution.tenant_id.clone();
    let response_id = execution.response_id.clone();
    if let Err(error) = state
        .durable
        .put_blob(
            BACKGROUND_RESPONSE_INPUTS_NAMESPACE,
            &tenant_id,
            &response_id,
            persisted_bytes,
        )
        .await
    {
        return internal_error_response(error);
    }
    let job = BackgroundResponseJob {
        tenant_id: tenant_id.clone(),
        response_id: response_id.clone(),
        created_at: execution.created_at,
        status: "queued".to_string(),
        cancel_requested: false,
        last_error: None,
    };
    if let Err(error) = state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            &tenant_id,
            &response_id,
            &job,
        )
        .await
    {
        return internal_error_response(error);
    }
    let response = openai_response_value_with_status(
        &execution,
        &execution.request_model,
        Vec::new(),
        String::new(),
        serde_json::Value::Null,
        "queued",
        None,
    );
    if let Err(error) = store_openai_response(
        &state.responses,
        &execution,
        &response,
        execution.conversation_messages.clone(),
        true,
    )
    .await
    {
        return internal_error_response(error);
    }
    spawn_background_response_job(state, job);
    Json(response).into_response()
}

fn spawn_background_response_job(state: AppState, job: BackgroundResponseJob) {
    let key = background_response_job_key(&job.tenant_id, &job.response_id);
    let runtime = state.jobs.clone();
    runtime.spawn(key, move |cancellation| async move {
        run_background_response_job(state, job, cancellation).await;
    });
}

async fn run_background_response_job(
    state: AppState,
    mut job: BackgroundResponseJob,
    cancellation: CancellationToken,
) {
    let latest = state
        .durable
        .get_json::<BackgroundResponseJob>(
            BACKGROUND_RESPONSES_NAMESPACE,
            &job.tenant_id,
            &job.response_id,
        )
        .await;
    if let Ok(Some(latest)) = latest {
        job = latest;
    }
    if job.cancel_requested || cancellation.is_cancelled() {
        finish_background_response_cancelled(&state, &mut job).await;
        return;
    }

    job.status = "in_progress".to_string();
    if state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            &job.tenant_id,
            &job.response_id,
            &job,
        )
        .await
        .is_err()
    {
        return;
    }
    if update_stored_background_response(&state, &job, "in_progress", None)
        .await
        .is_err()
    {
        return;
    }
    let bytes = match state
        .durable
        .get_blob(
            BACKGROUND_RESPONSE_INPUTS_NAMESPACE,
            &job.tenant_id,
            &job.response_id,
        )
        .await
    {
        Ok(Some(bytes)) => bytes,
        Ok(None) => {
            finish_background_response_failed(
                &state,
                &mut job,
                "background request payload is missing".to_string(),
            )
            .await;
            return;
        }
        Err(error) => {
            finish_background_response_failed(&state, &mut job, error).await;
            return;
        }
    };
    let raw_request = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(request) => request,
        Err(error) => {
            finish_background_response_failed(&state, &mut job, error.to_string()).await;
            return;
        }
    };
    let headers = match batch_api::tenant_headers_for_background(&job.tenant_id) {
        Ok(headers) => headers,
        Err(error) => {
            finish_background_response_failed(&state, &mut job, error).await;
            return;
        }
    };
    let execution = execute_openai_responses_value_with_id(
        &state,
        &headers,
        raw_request,
        true,
        Some(job.response_id.clone()),
    );
    tokio::pin!(execution);
    let result = tokio::select! {
        _ = cancellation.cancelled() => None,
        result = &mut execution => Some(result),
    };
    match result {
        None => finish_background_response_cancelled(&state, &mut job).await,
        Some(Ok(_)) => {
            job.status = "completed".to_string();
            job.last_error = None;
            if let Ok(Some(mut stored)) = state.responses.get(&job.tenant_id, &job.response_id) {
                stored.response["background"] = serde_json::Value::Bool(true);
                let _ = state.responses.put(stored).await;
            }
            let _ = state
                .durable
                .put_json(
                    BACKGROUND_RESPONSES_NAMESPACE,
                    &job.tenant_id,
                    &job.response_id,
                    &job,
                )
                .await;
        }
        Some(Err(error)) => {
            finish_background_response_failed(&state, &mut job, error).await;
        }
    }
}

async fn finish_background_response_cancelled(state: &AppState, job: &mut BackgroundResponseJob) {
    job.status = "cancelled".to_string();
    job.cancel_requested = true;
    let _ = update_stored_background_response(state, job, "cancelled", None).await;
    let _ = state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            &job.tenant_id,
            &job.response_id,
            job,
        )
        .await;
}

async fn finish_background_response_failed(
    state: &AppState,
    job: &mut BackgroundResponseJob,
    error: String,
) {
    job.status = "failed".to_string();
    job.last_error = Some(error.clone());
    let _ = update_stored_background_response(state, job, "failed", Some(error)).await;
    let _ = state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            &job.tenant_id,
            &job.response_id,
            job,
        )
        .await;
}

async fn update_stored_background_response(
    state: &AppState,
    job: &BackgroundResponseJob,
    status: &str,
    error: Option<String>,
) -> Result<(), String> {
    let Some(mut stored) = state.responses.get(&job.tenant_id, &job.response_id)? else {
        return Err(format!(
            "stored background response not found: {}",
            job.response_id
        ));
    };
    stored.response["status"] = serde_json::Value::String(status.to_string());
    stored.response["background"] = serde_json::Value::Bool(true);
    stored.response["error"] = error
        .map(|message| {
            serde_json::json!({
                "code": "background_execution_failed",
                "message": message
            })
        })
        .unwrap_or(serde_json::Value::Null);
    state.responses.put(stored).await
}

async fn recover_background_response_jobs(state: AppState) {
    let jobs = match state
        .durable
        .list_all_json::<BackgroundResponseJob>(BACKGROUND_RESPONSES_NAMESPACE)
        .await
    {
        Ok(jobs) => jobs,
        Err(_) => return,
    };
    for job in jobs {
        if matches!(job.status.as_str(), "queued" | "in_progress") {
            spawn_background_response_job(state.clone(), job);
        }
    }
}

fn background_response_job_key(tenant_id: &str, response_id: &str) -> String {
    format!("openai-response:{tenant_id}:{response_id}")
}

#[derive(Debug, Deserialize)]
struct ResponsesListQuery {
    limit: Option<usize>,
    after: Option<String>,
}

struct MaybeWebSocketUpgrade(Option<WebSocketUpgrade>);

impl<S> FromRequestParts<S> for MaybeWebSocketUpgrade
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            WebSocketUpgrade::from_request_parts(parts, state)
                .await
                .ok(),
        ))
    }
}

async fn responses_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ResponsesListQuery>,
    MaybeWebSocketUpgrade(websocket): MaybeWebSocketUpgrade,
) -> Response {
    if let Some(websocket) = websocket {
        return websocket
            .on_upgrade(move |socket| responses_websocket(socket, state, headers))
            .into_response();
    }

    responses_list_for_state(&state, &headers, query)
}

fn responses_list_for_state(
    state: &AppState,
    headers: &HeaderMap,
    query: ResponsesListQuery,
) -> Response {
    let tenant_id = RequestContext::from_headers(headers).tenant_id();
    let limit = query.limit.unwrap_or(100).clamp(1, 1000);
    let responses = match state.responses.list(tenant_id.as_ref()) {
        Ok(responses) => responses,
        Err(error) => return internal_error_response(error),
    };
    let mut seen_after = query.after.is_none();
    let data = responses
        .into_iter()
        .filter_map(|stored| {
            if !seen_after {
                seen_after = query.after.as_deref() == Some(stored.id.as_str());
                return None;
            }
            Some(stored.response)
        })
        .take(limit)
        .collect::<Vec<_>>();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
        "has_more": false
    }))
    .into_response()
}

async fn responses_websocket(mut socket: WebSocket, state: AppState, headers: HeaderMap) {
    while let Some(message) = socket.recv().await {
        let raw = match message {
            Ok(AxumWsMessage::Text(text)) => serde_json::from_str::<serde_json::Value>(&text),
            Ok(AxumWsMessage::Binary(bytes)) => serde_json::from_slice::<serde_json::Value>(&bytes),
            Ok(AxumWsMessage::Ping(payload)) => {
                if socket.send(AxumWsMessage::Pong(payload)).await.is_err() {
                    break;
                }
                continue;
            }
            Ok(AxumWsMessage::Pong(_)) => continue,
            Ok(AxumWsMessage::Close(_)) => break,
            Err(_) => break,
        };

        let events = match raw {
            Ok(raw) => {
                openai_responses_websocket_events_with_keepalive(&mut socket, &state, &headers, raw)
                    .await
            }
            Err(error) => Err(format!("invalid Responses WebSocket request: {error}")),
        }
        .unwrap_or_else(|error| {
            eprintln!("responses websocket error: {error}");
            vec![openai_responses_websocket_error_event(error)]
        });

        for event in events {
            if socket
                .send(AxumWsMessage::Text(event.to_string().into()))
                .await
                .is_err()
            {
                return;
            }
        }
    }
}

async fn openai_responses_websocket_events_with_keepalive(
    socket: &mut WebSocket,
    state: &AppState,
    headers: &HeaderMap,
    raw: serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    let events = openai_responses_websocket_events(state, headers, raw);
    tokio::pin!(events);
    let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(15));
    let mut keepalive_sequence = 1_000_u64;
    keepalive.tick().await;

    loop {
        tokio::select! {
            result = &mut events => return result,
            _ = keepalive.tick() => {
                socket
                    .send(AxumWsMessage::Text(
                        openai_responses_websocket_keepalive_event(keepalive_sequence)
                            .to_string()
                            .into(),
                    ))
                    .await
                    .map_err(|_| "Responses WebSocket closed while waiting for model output".to_string())?;
                keepalive_sequence = keepalive_sequence.saturating_add(1);
                socket
                    .send(AxumWsMessage::Ping(Vec::new().into()))
                    .await
                    .map_err(|_| "Responses WebSocket closed while waiting for model output".to_string())?;
            }
        }
    }
}

fn openai_responses_websocket_keepalive_event(sequence_number: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "response.in_progress",
        "sequence_number": sequence_number,
        "response": {
            "id": "resp_keepalive",
            "object": "response",
            "created_at": unix_timestamp_secs(),
            "status": "in_progress",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "metadata": {},
            "model": "",
            "output": [],
            "output_text": "",
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "store": false,
            "temperature": null,
            "tool_choice": "auto",
            "tools": [],
            "top_p": null,
            "truncation": "disabled",
            "usage": null
        }
    })
}

async fn openai_responses_websocket_events(
    state: &AppState,
    headers: &HeaderMap,
    mut raw: serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    let request_type = raw
        .get("type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "Responses WebSocket request must include type".to_string())?
        .to_string();

    match request_type.as_str() {
        "response.processed" => Ok(Vec::new()),
        "response.create" => {
            if let Some(object) = raw.as_object_mut() {
                object.remove("type");
            }
            let generate = raw
                .get("generate")
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            if let Some(object) = raw.as_object_mut() {
                object.remove("generate");
            }
            if !generate {
                let response =
                    empty_openai_responses_response_for_request(state, headers, raw).await?;
                return Ok(openai_responses_websocket_events_from_value(response));
            }

            let response = execute_openai_responses_value(state, headers, raw, true).await?;
            Ok(openai_responses_websocket_events_from_value(response))
        }
        other => Err(format!(
            "unsupported Responses WebSocket request type: {other}"
        )),
    }
}

async fn execute_openai_responses_value(
    state: &AppState,
    headers: &HeaderMap,
    raw_request: serde_json::Value,
    force_store_response: bool,
) -> Result<serde_json::Value, String> {
    execute_openai_responses_value_with_id(state, headers, raw_request, force_store_response, None)
        .await
}

async fn execute_openai_responses_value_with_id(
    state: &AppState,
    headers: &HeaderMap,
    raw_request: serde_json::Value,
    force_store_response: bool,
    forced_response_id: Option<String>,
) -> Result<serde_json::Value, String> {
    let request = parse_openai_responses_request(raw_request).map_err(api_error_message)?;
    let include_public_reasoning = state
        .public_reasoning_mode
        .resolve(openai_responses_public_reasoning_requested(&request));
    let request_context =
        RequestContext::from_headers(headers).with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let _tenant_permit = state
        .tenant_limiter
        .acquire(&tenant_id)
        .await
        .map_err(|error| error.to_string())?;
    let execution = prepare_openai_responses_execution_with_id(
        &state.responses,
        request,
        &request_context,
        forced_response_id,
    )
    .await
    .map_err(api_error_message)?;

    if matches!(
        openai_reasoning_effort(&execution.chat_request),
        Ok(ReasoningEffort::None)
    ) {
        let raw_chat_request = openai_chat_request_json(&execution.chat_request, false);
        let value = state
            .direct
            .openai_chat(raw_chat_request)
            .await
            .map_err(|error| error.to_string())?;
        let response = openai_responses_response_from_chat_completion(&execution, value.clone());
        store_openai_response(
            &state.responses,
            &execution,
            &response,
            response_conversation_messages_from_chat_completion(
                &execution,
                execution.conversation_messages.clone(),
                &value,
            ),
            force_store_response,
        )
        .await?;
        let telemetry_context = direct_telemetry_context(
            "openai.responses",
            execution.request_model.clone(),
            "openai_responses",
            &request_context,
            true,
            None,
        );
        emit_direct_telemetry(&telemetry_context, response_usage(&value));
        return Ok(response);
    }

    let model = execution.request_model.clone();
    let mut normalized =
        normalize_openai_chat_with_context(execution.chat_request.clone(), &request_context)
            .map_err(api_error_message)?;
    normalized.public_reasoning_enabled = include_public_reasoning;
    let prepared_context = state.context.prepare(&mut normalized).await?;
    let telemetry_context =
        telemetry_context_from_normalized("openai.responses", &normalized, true, None);
    let training_request = state.training_trace.capture_request(&normalized);

    let output = state
        .kernel
        .run(normalized)
        .await
        .map_err(|error| error.to_string())?;
    let context_report = finalize_context_report(&state.context, &prepared_context, &output)
        .await
        .map_err(|error| error.to_string())?;
    emit_kernel_telemetry(&telemetry_context, &output, context_report.as_ref());
    if let Err(error) = state
        .training_trace
        .record_kernel(training_request.as_ref(), &output)
    {
        log_training_trace_error(error);
    }
    let conversation_messages = response_conversation_messages_from_kernel_output(
        &execution,
        execution.conversation_messages.clone(),
        &output,
    );
    let mut response =
        format_openai_responses_response(&execution, &model, output, include_public_reasoning);
    attach_context_report(&mut response, prepared_context.as_ref(), context_report);
    store_openai_response(
        &state.responses,
        &execution,
        &response,
        conversation_messages,
        force_store_response,
    )
    .await?;
    Ok(response)
}

fn api_error_message(error: ApiError) -> String {
    match error {
        ApiError::InvalidRequest(message) => message,
        ApiError::StreamUnsupported => "streaming is handled by the gateway transport".to_string(),
    }
}

async fn empty_openai_responses_response_for_request(
    state: &AppState,
    headers: &HeaderMap,
    raw_request: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let request = parse_openai_responses_request(raw_request).map_err(api_error_message)?;
    let request_context =
        RequestContext::from_headers(headers).with_metadata_overrides(&request.metadata);
    let tenant_id = request_context.tenant_id();
    let _tenant_permit = state
        .tenant_limiter
        .acquire(&tenant_id)
        .await
        .map_err(|error| error.to_string())?;
    let execution = prepare_openai_responses_execution(&state.responses, request, &request_context)
        .await
        .map_err(api_error_message)?;
    let response = openai_response_value(
        &execution,
        &execution.request_model,
        Vec::new(),
        String::new(),
        serde_json::json!({
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0
        }),
    );
    store_openai_response(
        &state.responses,
        &execution,
        &response,
        execution.conversation_messages.clone(),
        true,
    )
    .await?;
    Ok(response)
}

fn openai_responses_websocket_events_from_value(
    response: serde_json::Value,
) -> Vec<serde_json::Value> {
    let mut events = vec![serde_json::json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": response_with_status(response.clone(), "in_progress")
    })];
    events.extend(openai_responses_output_event_values(&response, 1));
    events.push(serde_json::json!({
        "type": "response.completed",
        "sequence_number": 10_000,
        "response": response
    }));
    events
}

fn openai_responses_websocket_error_event(message: String) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "status_code": 500,
        "error": {
            "message": message,
            "type": "server_error",
            "code": "kernel_error"
        }
    })
}

async fn responses_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    match state.responses.get(tenant_id.as_ref(), &response_id) {
        Ok(Some(stored)) => Json(stored.response).into_response(),
        Ok(None) => not_found_response(format!("response not found: {response_id}")),
        Err(error) => internal_error_response(error),
    }
}

async fn responses_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let background_job = match state
        .durable
        .get_json::<BackgroundResponseJob>(
            BACKGROUND_RESPONSES_NAMESPACE,
            tenant_id.as_ref(),
            &response_id,
        )
        .await
    {
        Ok(job) => job,
        Err(error) => return internal_error_response(error),
    };
    if background_job
        .as_ref()
        .is_some_and(|job| matches!(job.status.as_str(), "queued" | "in_progress" | "cancelling"))
    {
        return conflict_response(
            "cancel an active background response before deleting it".to_string(),
        );
    }
    match state
        .responses
        .delete(tenant_id.as_ref(), &response_id)
        .await
    {
        Ok(true) => {
            if background_job.is_some() {
                let _ = state
                    .durable
                    .delete_json(
                        BACKGROUND_RESPONSES_NAMESPACE,
                        tenant_id.as_ref(),
                        &response_id,
                    )
                    .await;
                let _ = state
                    .durable
                    .delete_blob(
                        BACKGROUND_RESPONSE_INPUTS_NAMESPACE,
                        tenant_id.as_ref(),
                        &response_id,
                    )
                    .await;
            }
            Json(serde_json::json!({
                "id": response_id,
                "object": "response.deleted",
                "deleted": true
            }))
            .into_response()
        }
        Ok(false) => not_found_response(format!("response not found: {response_id}")),
        Err(error) => internal_error_response(error),
    }
}

async fn responses_input_items(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    match state.responses.get(tenant_id.as_ref(), &response_id) {
        Ok(Some(stored)) => Json(serde_json::json!({
            "object": "list",
            "data": stored.input_items,
            "has_more": false
        }))
        .into_response(),
        Ok(None) => not_found_response(format!("response not found: {response_id}")),
        Err(error) => internal_error_response(error),
    }
}

async fn responses_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut job = match state
        .durable
        .get_json::<BackgroundResponseJob>(
            BACKGROUND_RESPONSES_NAMESPACE,
            tenant_id.as_ref(),
            &response_id,
        )
        .await
    {
        Ok(Some(job)) => job,
        Ok(None) => {
            return match state.responses.get(tenant_id.as_ref(), &response_id) {
                Ok(Some(_)) => api_error_response(ApiError::InvalidRequest(
                    "only Responses created with background=true can be cancelled".to_string(),
                )),
                Ok(None) => not_found_response(format!("response not found: {response_id}")),
                Err(error) => internal_error_response(error),
            };
        }
        Err(error) => return internal_error_response(error),
    };
    if matches!(job.status.as_str(), "completed" | "failed" | "cancelled") {
        return api_error_response(ApiError::InvalidRequest(format!(
            "background response {response_id} is already {}",
            job.status
        )));
    }
    job.cancel_requested = true;
    job.status = "cancelling".to_string();
    if let Err(error) = state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            tenant_id.as_ref(),
            &response_id,
            &job,
        )
        .await
    {
        return internal_error_response(error);
    }
    state.jobs.cancel(&background_response_job_key(
        tenant_id.as_ref(),
        &response_id,
    ));
    job.status = "cancelled".to_string();
    if let Err(error) = update_stored_background_response(&state, &job, "cancelled", None).await {
        return internal_error_response(error);
    }
    if let Err(error) = state
        .durable
        .put_json(
            BACKGROUND_RESPONSES_NAMESPACE,
            tenant_id.as_ref(),
            &response_id,
            &job,
        )
        .await
    {
        return internal_error_response(error);
    }
    match state.responses.get(tenant_id.as_ref(), &response_id) {
        Ok(Some(stored)) => Json(stored.response).into_response(),
        Ok(None) => not_found_response(format!("response not found: {response_id}")),
        Err(error) => internal_error_response(error),
    }
}

async fn responses_input_tokens(Json(raw): Json<serde_json::Value>) -> Response {
    let request = match parse_openai_responses_request(raw) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let token_estimate = estimate_response_input_tokens(&request.input)
        + request
            .instructions
            .as_str()
            .map(estimate_text_tokens)
            .unwrap_or_default();
    Json(serde_json::json!({
        "object": "response.input_tokens",
        "input_tokens": token_estimate
    }))
    .into_response()
}

async fn responses_compact(Json(raw): Json<serde_json::Value>) -> Response {
    let input = raw.get("input").cloned().unwrap_or(serde_json::Value::Null);
    let text = response_input_text_for_compaction(&input);
    let encrypted_content = format!("sha256:{}", sha256_hex(&text));
    Json(serde_json::json!({
        "id": format!("compaction_{}", Uuid::new_v4()),
        "object": "response.compaction",
        "created_at": unix_timestamp_secs(),
        "output": [{
            "id": format!("ci_{}", Uuid::new_v4()),
            "type": "compaction",
            "encrypted_content": encrypted_content,
            "created_by": "multi-agent-api"
        }],
        "usage": openai_responses_usage_json(&ProviderUsage {
            input_tokens: estimate_text_tokens(&text),
            output_tokens: 0
        })
    }))
    .into_response()
}

async fn messages_count_tokens(Json(raw_request): Json<serde_json::Value>) -> Response {
    if let Err(error) = parse_anthropic_request(raw_request.clone()) {
        return api_error_response(error);
    }

    let system_text = response_input_text_for_compaction(
        raw_request
            .get("system")
            .unwrap_or(&serde_json::Value::Null),
    );
    let message_text = response_input_text_for_compaction(
        raw_request
            .get("messages")
            .unwrap_or(&serde_json::Value::Null),
    );
    let tool_text = raw_request
        .get("tools")
        .filter(|tools| !tools.is_null())
        .map(serde_json::Value::to_string)
        .unwrap_or_default();
    let image_tokens = count_json_blocks_of_type(
        raw_request
            .get("messages")
            .unwrap_or(&serde_json::Value::Null),
        "image",
    )
    .saturating_mul(256);
    let input_tokens = estimate_text_tokens(&system_text)
        .saturating_add(estimate_text_tokens(&message_text))
        .saturating_add(estimate_text_tokens(&tool_text))
        .saturating_add(image_tokens);

    Json(serde_json::json!({"input_tokens": input_tokens})).into_response()
}

fn count_json_blocks_of_type(value: &serde_json::Value, expected_type: &str) -> u32 {
    match value {
        serde_json::Value::Array(values) => values.iter().fold(0_u32, |count, value| {
            count.saturating_add(count_json_blocks_of_type(value, expected_type))
        }),
        serde_json::Value::Object(object) => {
            let own = u32::from(
                object.get("type").and_then(serde_json::Value::as_str) == Some(expected_type),
            );
            object.values().fold(own, |count, value| {
                count.saturating_add(count_json_blocks_of_type(value, expected_type))
            })
        }
        _ => 0,
    }
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
        Err(error) => return provider_error_response(error),
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
                Err(error) => provider_error_response(error),
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
            Err(error) => provider_error_response(error),
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
            Err(error) => kernel_error_response(error),
        };
    }

    let metrics = state.metrics.clone();
    let response_future = async move {
        let _tenant_permit = tenant_permit;
        match state.kernel.run(normalized).await {
            Ok(output) => {
                let context_report =
                    match finalize_context_report(&state.context, &prepared_context, &output).await
                    {
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
                    return format_anthropic_stream_response(
                        model,
                        output,
                        include_public_reasoning,
                    );
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
            Err(error) => kernel_error_response(error),
        }
    };
    if stream {
        orchestration_sse_response(response_future, metrics)
    } else {
        response_future.await
    }
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
            NormalizedContentPart::ProviderContent { value, .. } => value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("[provider_content]"),
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
            "description": "Dispatch bounded model-selected sub-agent tasks during commercial orchestration.",
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

fn kernel_error_response(error: KernelError) -> Response {
    if let KernelError::Provider(provider_error) = error {
        return provider_error_response(provider_error);
    }
    let (status, code, error_type) = match &error {
        KernelError::RequestTimeout { .. } | KernelError::AgentTimeout { .. } => (
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "timeout_error",
        ),
        KernelError::ProviderRejected(message)
            if message.contains("provider queue wait exceeded") =>
        {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "provider_overloaded",
                "server_error",
            )
        }
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "kernel_error",
            "server_error",
        ),
    };
    (
        status,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message: error.to_string(),
                r#type: error_type.to_string(),
                code: code.to_string(),
            },
        }),
    )
        .into_response()
}

fn provider_error_response(error: ProviderError) -> Response {
    let retry_after_ms = error.retry_after_ms();
    let (status, code, error_type) = match &error {
        ProviderError::Http { status, code, .. } => {
            let status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
            let error_type = if status == StatusCode::TOO_MANY_REQUESTS {
                "rate_limit_error"
            } else if status.is_client_error() {
                "invalid_request_error"
            } else {
                "server_error"
            };
            (
                status,
                code.clone().unwrap_or_else(|| "upstream_error".to_string()),
                error_type,
            )
        }
        ProviderError::CircuitOpen { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "provider_circuit_open".to_string(),
            "server_error",
        ),
        ProviderError::QueueTimeout { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "provider_overloaded".to_string(),
            "server_error",
        ),
        ProviderError::Transport { .. } => (
            StatusCode::BAD_GATEWAY,
            "provider_transport_error".to_string(),
            "server_error",
        ),
        ProviderError::InvalidResponse { .. } => (
            StatusCode::BAD_GATEWAY,
            "invalid_provider_response".to_string(),
            "server_error",
        ),
        ProviderError::Rejected(_) => (
            StatusCode::BAD_GATEWAY,
            "provider_rejected".to_string(),
            "server_error",
        ),
    };
    let mut response = (
        status,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message: error.to_string(),
                r#type: error_type.to_string(),
                code,
            },
        }),
    )
        .into_response();
    if let Some(retry_after_ms) = retry_after_ms {
        let retry_after_seconds = retry_after_ms.saturating_add(999) / 1_000;
        if let Ok(value) = header::HeaderValue::from_str(&retry_after_seconds.max(1).to_string()) {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
    }
    response
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

fn conflict_response(message: String) -> Response {
    (
        StatusCode::CONFLICT,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message,
                r#type: "conflict_error".to_string(),
                code: "conflict".to_string(),
            },
        }),
    )
        .into_response()
}

fn not_found_response(message: String) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(CompatibilityError {
            error: CompatibilityErrorBody {
                message,
                r#type: "not_found_error".to_string(),
                code: "not_found".to_string(),
            },
        }),
    )
        .into_response()
}

fn log_training_trace_error(error: String) {
    eprintln!("training trace error: {error}");
}

fn response_with_provider_permit(
    mut response: Response,
    permit: ProviderAdmissionPermit,
) -> Response {
    response.extensions_mut().insert(Arc::new(permit));
    response
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
        .header("x-accel-buffering", "no")
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
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|error| internal_error_response(error.to_string()))
}

fn orchestration_stream_heartbeat_interval() -> Duration {
    let seconds = std::env::var("MIYA_STREAM_HEARTBEAT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10);
    Duration::from_secs(seconds)
}

fn orchestration_sse_response<F>(future: F, metrics: RuntimeMetrics) -> Response
where
    F: Future<Output = Response> + Send + 'static,
{
    orchestration_sse_response_with_interval(
        future,
        metrics,
        orchestration_stream_heartbeat_interval(),
    )
}

fn orchestration_sse_response_with_interval<F>(
    future: F,
    metrics: RuntimeMetrics,
    heartbeat: Duration,
) -> Response
where
    F: Future<Output = Response> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    metrics
        .orchestration_streams_active
        .fetch_add(1, Ordering::Relaxed);
    let task_metrics = metrics.clone();
    tokio::spawn(async move {
        let run = async {
            if tx
                .send(Ok(Bytes::from_static(b": miya orchestration active\n\n")))
                .await
                .is_err()
            {
                return;
            }

            let mut ticker = tokio::time::interval(heartbeat);
            ticker.tick().await;
            tokio::pin!(future);
            loop {
                tokio::select! {
                    response = &mut future => {
                        let status = response.status();
                        let is_sse = response
                            .headers()
                            .get(header::CONTENT_TYPE)
                            .and_then(|value| value.to_str().ok())
                            .is_some_and(|value| value.starts_with("text/event-stream"));
                        let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
                            Ok(body) => body,
                            Err(error) => Bytes::from(
                                sse_event(
                                    "error",
                                    serde_json::json!({
                                        "type": "error",
                                        "error": {
                                            "type": "stream_body_error",
                                            "message": error.to_string()
                                        }
                                    }),
                                ),
                            ),
                        };
                        let chunk = if status.is_success() && is_sse {
                            body
                        } else {
                            let value = serde_json::from_slice::<serde_json::Value>(&body)
                                .unwrap_or_else(|_| serde_json::json!({
                                    "error": {
                                        "type": "server_error",
                                        "message": String::from_utf8_lossy(&body)
                                    }
                                }));
                            Bytes::from(sse_event(
                                "error",
                                serde_json::json!({
                                    "type": "error",
                                    "error": value.get("error").cloned().unwrap_or(value)
                                }),
                            ))
                        };
                        let _ = tx.send(Ok(chunk)).await;
                        return;
                    }
                    _ = ticker.tick() => {
                        match tx.try_send(Ok(Bytes::from_static(b": miya orchestration active\n\n"))) {
                            Ok(()) => {
                                task_metrics
                                    .orchestration_stream_heartbeats
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            Err(mpsc::error::TrySendError::Full(_)) => {}
                            Err(mpsc::error::TrySendError::Closed(_)) => return,
                        }
                    }
                }
            }
        };
        run.await;
        task_metrics
            .orchestration_streams_active
            .fetch_sub(1, Ordering::Relaxed);
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    sse_stream_response(stream)
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

    if let Some((reason, children)) = output.trace_events.iter().rev().find_map(|event| {
        let KernelTraceEvent::SpawnPlan {
            reason, children, ..
        } = event
        else {
            return None;
        };
        Some((reason, children))
    }) {
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
        if verification.passed {
            "passed"
        } else if verification.unresolved_tool_calls.is_empty() {
            "completed with semantic verification issues"
        } else {
            "requires client tool action"
        },
        verification.issues.len(),
        verification.unresolved_tool_calls.len(),
        output.provider_call_count,
        verification.budget_summary.token_used,
        verification.budget_summary.token_budget
    ));

    if !output.encrypted_subagent_state.is_empty() {
        lines.push(
            "- Raw sub-agent state remains encrypted; public reasoning contains sanitized model-output summaries only."
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
        truncate_summary_text(&sanitize_public_agent_summary_text(line), 280)
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
    let sanitized = sanitize_public_agent_summary_text(text);
    let compact = compact_summary_whitespace(&sanitized);
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

fn sanitize_public_agent_summary_text(text: &str) -> String {
    let without_hidden = strip_public_hidden_reasoning_markers(text);
    let mut kept = Vec::new();
    let mut redacted_tool_line = false;

    for line in without_hidden.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if looks_like_child_tool_payload(trimmed) {
            redacted_tool_line = true;
            continue;
        }
        kept.push(trimmed.to_string());
    }

    if kept.is_empty() && redacted_tool_line {
        "reported an internal tool attempt without a public textual finding".to_string()
    } else {
        kept.join(" ")
    }
}

fn strip_public_hidden_reasoning_markers(text: &str) -> String {
    let mut cleaned = text
        .replace("<think>", " ")
        .replace("</think>", " ")
        .replace("<|channel>thought", " ")
        .replace("<channel|>", " ")
        .replace("</antthinking>", " ")
        .replace("</antthinkin>", " ")
        .replace("<antthinking>", " ")
        .replace("<antthinkin>", " ");

    cleaned = strip_between_markers(&cleaned, "<tool_call>", "</tool_call>");
    cleaned = strip_between_markers(&cleaned, "<tool_code>", "</tool_code>");
    cleaned
}

fn strip_between_markers(text: &str, start: &str, end: &str) -> String {
    let mut remaining = text.to_string();
    while let Some(start_index) = remaining.find(start) {
        let after_start = start_index + start.len();
        if let Some(end_offset) = remaining[after_start..].find(end) {
            let end_index = after_start + end_offset + end.len();
            remaining.replace_range(start_index..end_index, " ");
        } else {
            remaining.replace_range(start_index..after_start, " ");
        }
    }
    remaining
}

fn looks_like_child_tool_payload(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("tool call")
        || lower.contains("<tool_call")
        || lower.contains("<tool_code")
        || lower.contains("<command>")
        || lower.contains("<command_file>")
        || lower.contains("<arg_cmd>")
        || lower.contains("exec_command")
        || lower.contains("write_file")
        || lower.contains("\"cmd\"")
        || lower.contains("\"command\"")
        || lower.contains("{\"cmd\"")
        || lower.contains("{\"command\"")
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
    tool_names: &BTreeSet<String>,
    include_encrypted_state: bool,
    tool_response_format: OpenAiToolResponseFormat,
    include_public_reasoning: bool,
) -> serde_json::Value {
    let encrypted_state = output.encrypted_subagent_state.clone();
    let usage = openai_usage_json(&output.usage);
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));
    let text_tool_call = output
        .verification
        .passed
        .then(|| parse_text_tool_call(&output.final_text, tool_names))
        .flatten()
        .map(parsed_text_tool_call_record);
    let mut value = if let Some(call) = text_tool_call {
        if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
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
                        "function_call": openai_legacy_function_call_json(call)
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
                        "tool_calls": [openai_tool_call_json(call)]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": usage
            })
        }
    } else if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
    {
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

#[derive(Clone)]
struct OpenAiResponsesExecution {
    response_id: String,
    created_at: u64,
    tenant_id: String,
    request_model: String,
    previous_response_id: Option<String>,
    store: bool,
    background: bool,
    metadata: serde_json::Value,
    tools: Vec<serde_json::Value>,
    tool_choice: serde_json::Value,
    tool_kinds: BTreeMap<String, String>,
    parallel_tool_calls: bool,
    instructions: serde_json::Value,
    input_items: Vec<serde_json::Value>,
    conversation_messages: Vec<OpenAiMessage>,
    chat_request: OpenAiChatRequest,
}

async fn prepare_openai_responses_execution(
    store: &ResponsesStore,
    request: OpenAiResponsesRequest,
    request_context: &RequestContext,
) -> Result<OpenAiResponsesExecution, ApiError> {
    prepare_openai_responses_execution_with_id(store, request, request_context, None).await
}

async fn prepare_openai_responses_execution_with_id(
    store: &ResponsesStore,
    request: OpenAiResponsesRequest,
    request_context: &RequestContext,
    forced_response_id: Option<String>,
) -> Result<OpenAiResponsesExecution, ApiError> {
    validate_openai_responses_request(&request)?;
    let tenant_id = request_context.tenant_id();
    let previous_messages = if let Some(previous_response_id) = &request.previous_response_id {
        let stored = store
            .get(tenant_id.as_ref(), previous_response_id)
            .map_err(ApiError::InvalidRequest)?
            .ok_or_else(|| {
                ApiError::InvalidRequest(format!(
                    "previous_response_id not found: {previous_response_id}"
                ))
            })?;
        stored.conversation_messages
    } else {
        Vec::new()
    };

    let instruction_messages = response_instructions_to_messages(&request.instructions)?;
    let current_messages = response_input_to_messages(&request.input)?;
    let mut chat_messages = instruction_messages;
    chat_messages.extend(previous_messages.clone());
    chat_messages.extend(current_messages.clone());

    let mut conversation_messages = previous_messages;
    conversation_messages.extend(current_messages);

    let (mut tools, mut tool_kinds) = response_tools_to_openai_tools(&request.tools);
    let tool_choice =
        response_tool_choice_to_openai_chat(&request.tool_choice, &mut tools, &mut tool_kinds);
    let reasoning = reasoning_with_top_level_effort(&request.reasoning, &request.extra);
    let mut extra = responses_provider_options(&request);
    let chat_template_kwargs = extra
        .remove("chat_template_kwargs")
        .unwrap_or(serde_json::Value::Null);
    let enable_thinking = extra.remove("enable_thinking").and_then(json_bool);
    let preserve_thinking = extra.remove("preserve_thinking").and_then(json_bool);
    let chat_request = OpenAiChatRequest {
        model: request.model.clone(),
        messages: chat_messages,
        tools,
        tool_choice,
        functions: Vec::new(),
        function_call: serde_json::Value::Null,
        parallel_tool_calls: request.parallel_tool_calls,
        thinking: request.thinking.clone(),
        reasoning,
        chat_template_kwargs,
        enable_thinking,
        preserve_thinking,
        stream: false,
        metadata: request.metadata.clone(),
        extra,
    };

    Ok(OpenAiResponsesExecution {
        response_id: forced_response_id.unwrap_or_else(|| format!("resp_{}", Uuid::new_v4())),
        created_at: unix_timestamp_secs(),
        tenant_id: tenant_id.as_ref().to_string(),
        request_model: request.model,
        previous_response_id: request.previous_response_id,
        store: request.store.unwrap_or(true),
        background: request.background,
        metadata: request.metadata,
        tools: request.tools,
        tool_choice: request.tool_choice,
        tool_kinds,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        instructions: request.instructions,
        input_items: response_input_items(&request.input),
        conversation_messages,
        chat_request,
    })
}

fn validate_openai_responses_request(request: &OpenAiResponsesRequest) -> Result<(), ApiError> {
    if request.background && request.stream {
        return Err(ApiError::InvalidRequest(
            "background Responses cannot use stream=true; poll the stored response instead"
                .to_string(),
        ));
    }
    for field in ["conversation", "prompt"] {
        if request
            .extra
            .get(field)
            .is_some_and(|value| !value.is_null())
        {
            return Err(ApiError::InvalidRequest(format!(
                "Responses field {field} is not implemented by this gateway"
            )));
        }
    }
    if let Some(truncation) = request.extra.get("truncation")
        && !truncation.is_null()
        && truncation.as_str() != Some("disabled")
    {
        return Err(ApiError::InvalidRequest(
            "only truncation=disabled is currently supported".to_string(),
        ));
    }
    Ok(())
}

fn responses_provider_options(
    request: &OpenAiResponsesRequest,
) -> BTreeMap<String, serde_json::Value> {
    let mut extra = BTreeMap::new();
    for (key, value) in &request.extra {
        if !is_responses_only_option(key) && !is_gateway_extra_option(key) {
            extra.insert(key.clone(), value.clone());
        }
    }

    if let Some(max_output_tokens) = request.max_output_tokens {
        extra
            .entry("max_tokens".to_string())
            .or_insert_with(|| serde_json::json!(max_output_tokens));
    }
    if let Some(temperature) = request.temperature {
        extra.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        extra.insert("top_p".to_string(), serde_json::json!(top_p));
    }
    if let Some(top_logprobs) = request.top_logprobs {
        extra.insert("top_logprobs".to_string(), serde_json::json!(top_logprobs));
    }
    if let Some(response_format) = responses_text_to_chat_response_format(&request.extra) {
        extra.insert("response_format".to_string(), response_format);
    }
    if let Some(verbosity) = request
        .extra
        .get("text")
        .and_then(|text| text.get("verbosity"))
        .filter(|verbosity| !verbosity.is_null())
    {
        extra.insert("verbosity".to_string(), verbosity.clone());
    }
    extra
}

fn is_responses_only_option(key: &str) -> bool {
    matches!(
        key,
        "background"
            | "client_metadata"
            | "context_management"
            | "conversation"
            | "generate"
            | "include"
            | "max_tool_calls"
            | "previous_response_id"
            | "prompt"
            | "store"
            | "stream_options"
            | "text"
            | "truncation"
            | "input"
            | "instructions"
    )
}

fn responses_text_to_chat_response_format(
    extra: &BTreeMap<String, serde_json::Value>,
) -> Option<serde_json::Value> {
    let format = extra.get("text")?.get("format")?;
    match format.get("type").and_then(|value| value.as_str()) {
        Some("json_object") => Some(serde_json::json!({"type": "json_object"})),
        Some("json_schema") => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": format.get("schema").cloned().unwrap_or_else(|| serde_json::json!({}))
        })),
        _ => None,
    }
}

fn response_instructions_to_messages(
    instructions: &serde_json::Value,
) -> Result<Vec<OpenAiMessage>, ApiError> {
    if instructions.is_null() {
        return Ok(Vec::new());
    }
    if let Some(text) = instructions.as_str() {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        return Ok(vec![openai_text_message("system", text.to_string())]);
    }
    if instructions.is_array() || instructions.is_object() {
        let mut messages = response_input_to_messages(instructions)?;
        for message in &mut messages {
            if message.role == "developer" {
                message.role = "system".to_string();
            }
        }
        return Ok(messages);
    }
    Err(ApiError::InvalidRequest(
        "Responses instructions must be a string, input item, or input item array".to_string(),
    ))
}

fn response_input_items(input: &serde_json::Value) -> Vec<serde_json::Value> {
    if input.is_null() {
        Vec::new()
    } else if let Some(text) = input.as_str() {
        vec![serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        })]
    } else if let Some(items) = input.as_array() {
        items.clone()
    } else {
        vec![input.clone()]
    }
}

fn response_input_to_messages(input: &serde_json::Value) -> Result<Vec<OpenAiMessage>, ApiError> {
    if input.is_null() {
        return Ok(Vec::new());
    }
    if let Some(text) = input.as_str() {
        return Ok(vec![openai_text_message("user", text.to_string())]);
    }
    if let Some(items) = input.as_array() {
        let mut messages = Vec::new();
        for item in items {
            messages.extend(response_input_item_to_messages(item)?);
        }
        return Ok(messages);
    }
    if input.is_object() {
        return response_input_item_to_messages(input);
    }
    Err(ApiError::InvalidRequest(
        "Responses input must be a string, object, or array".to_string(),
    ))
}

fn response_input_item_to_messages(
    item: &serde_json::Value,
) -> Result<Vec<OpenAiMessage>, ApiError> {
    let Some(object) = item.as_object() else {
        return Err(ApiError::InvalidRequest(
            "Responses input array items must be objects".to_string(),
        ));
    };
    let item_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("message");
    match item_type {
        "input_audio" | "input_file" | "input_image" | "input_text" => Ok(vec![OpenAiMessage {
            role: "user".to_string(),
            content: response_message_content_to_openai_content(Some(&serde_json::Value::Array(
                vec![item.clone()],
            ))),
            tool_calls: Vec::new(),
            function_call: None,
            tool_call_id: None,
            name: None,
        }]),
        "message" => {
            let role = object
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("user");
            let role = if role == "developer" { "system" } else { role };
            Ok(vec![OpenAiMessage {
                role: role.to_string(),
                content: response_message_content_to_openai_content(object.get("content")),
                tool_calls: Vec::new(),
                function_call: None,
                tool_call_id: None,
                name: None,
            }])
        }
        "function_call_output"
        | "custom_tool_call_output"
        | "local_shell_call_output"
        | "shell_call_output"
        | "apply_patch_call_output"
        | "computer_call_output" => {
            let call_id = response_tool_output_call_id(item).ok_or_else(|| {
                ApiError::InvalidRequest(format!("{item_type} must include call_id or id"))
            })?;
            Ok(vec![OpenAiMessage {
                role: "tool".to_string(),
                content: Some(OpenAiContent::Text(
                    response_tool_output_text(item).unwrap_or_default(),
                )),
                tool_calls: Vec::new(),
                function_call: None,
                tool_call_id: Some(call_id),
                name: None,
            }])
        }
        "function_call" | "custom_tool_call" | "local_shell_call" | "shell_call"
        | "apply_patch_call" | "mcp_call" => Ok(vec![response_tool_call_item_to_message(item)]),
        "reasoning" | "compaction" | "item_reference" => Ok(Vec::new()),
        _ => Ok(vec![openai_text_message("user", item.to_string())]),
    }
}

fn response_message_content_to_openai_content(
    content: Option<&serde_json::Value>,
) -> Option<OpenAiContent> {
    let content = content?;
    if let Some(text) = content.as_str() {
        return Some(OpenAiContent::Text(text.to_string()));
    }
    let parts = content.as_array()?;
    let mut openai_parts = Vec::new();
    for part in parts {
        let Some(part_type) = part.get("type").and_then(|value| value.as_str()) else {
            continue;
        };
        match part_type {
            "input_text" | "output_text" | "text" => {
                if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                    openai_parts.push(OpenAiContentPart::Text {
                        text: text.to_string(),
                    });
                }
            }
            "input_image" | "image_url" => {
                let url = part
                    .get("image_url")
                    .and_then(|value| {
                        value.as_str().map(str::to_string).or_else(|| {
                            value
                                .get("url")
                                .and_then(|url| url.as_str())
                                .map(str::to_string)
                        })
                    })
                    .or_else(|| {
                        part.get("url")
                            .and_then(|url| url.as_str())
                            .map(str::to_string)
                    });
                if let Some(url) = url {
                    openai_parts.push(OpenAiContentPart::ImageUrl {
                        image_url: OpenAiImageUrl { url },
                    });
                }
            }
            "input_audio" => {
                if let Some(input_audio) = part.get("input_audio") {
                    openai_parts.push(OpenAiContentPart::InputAudio {
                        input_audio: input_audio.clone(),
                        extra: BTreeMap::new(),
                    });
                }
            }
            "input_file" | "file" => {
                let file = if let Some(file) = part.get("file") {
                    file.clone()
                } else {
                    let mut file = serde_json::Map::new();
                    for key in ["file_data", "file_id", "file_url", "filename", "detail"] {
                        if let Some(value) = part.get(key) {
                            file.insert(key.to_string(), value.clone());
                        }
                    }
                    serde_json::Value::Object(file)
                };
                openai_parts.push(OpenAiContentPart::File {
                    file,
                    extra: BTreeMap::new(),
                });
            }
            "refusal" => {
                if let Some(refusal) = part.get("refusal").and_then(serde_json::Value::as_str) {
                    openai_parts.push(OpenAiContentPart::Refusal {
                        refusal: refusal.to_string(),
                    });
                }
            }
            _ => {}
        }
    }
    if openai_parts.is_empty() {
        None
    } else {
        Some(OpenAiContent::Parts(openai_parts))
    }
}

fn response_tool_output_call_id(item: &serde_json::Value) -> Option<String> {
    item.get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn response_tool_output_text(item: &serde_json::Value) -> Option<String> {
    let output = item.get("output")?;
    output
        .as_str()
        .map(str::to_string)
        .or_else(|| Some(output.to_string()))
}

fn response_tool_call_item_to_message(item: &serde_json::Value) -> OpenAiMessage {
    let item_type = item
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("function_call");
    let tool_name = match item_type {
        "custom_tool_call" => item
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("custom"),
        "local_shell_call" => "local_shell",
        "shell_call" => "shell",
        "apply_patch_call" => "apply_patch",
        "mcp_call" => item
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("mcp_call"),
        _ => item
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("function_call"),
    };
    let arguments = match item_type {
        "function_call" => item.get("arguments").cloned().unwrap_or_default(),
        "custom_tool_call" => item.get("input").cloned().unwrap_or_default(),
        "local_shell_call" | "shell_call" => item.get("action").cloned().unwrap_or_default(),
        "apply_patch_call" => item.get("operation").cloned().unwrap_or_default(),
        "mcp_call" => item.get("arguments").cloned().unwrap_or_default(),
        _ => serde_json::Value::Null,
    };
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|value| value.as_str())
        .unwrap_or("call-unknown")
        .to_string();
    OpenAiMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: vec![OpenAiMessageToolCall {
            id: call_id,
            r#type: "function".to_string(),
            function: OpenAiMessageToolCallFunction {
                name: tool_name.to_string(),
                arguments,
            },
        }],
        function_call: None,
        tool_call_id: None,
        name: None,
    }
}

fn response_tools_to_openai_tools(
    tools: &[serde_json::Value],
) -> (Vec<OpenAiTool>, BTreeMap<String, String>) {
    let mut openai_tools = Vec::new();
    let mut tool_kinds = BTreeMap::new();
    for tool in tools {
        if let Some((openai_tool, response_kind)) = response_tool_to_openai_tool(tool) {
            tool_kinds.insert(openai_tool.function.name.clone(), response_kind);
            openai_tools.push(openai_tool);
        }
    }
    (openai_tools, tool_kinds)
}

fn response_tool_to_openai_tool(tool: &serde_json::Value) -> Option<(OpenAiTool, String)> {
    let object = tool.as_object()?;
    let kind = object
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("function");
    let function = object.get("function").and_then(|value| value.as_object());
    let raw_name = function
        .and_then(|function| function.get("name"))
        .or_else(|| object.get("name"))
        .or_else(|| object.get("server_label"))
        .and_then(|value| value.as_str())
        .unwrap_or(kind);
    let name = sanitize_tool_name(match kind {
        "local_shell" => "local_shell",
        "shell" => "shell",
        "apply_patch" => "apply_patch",
        _ => raw_name,
    });
    if name.is_empty() {
        return None;
    }
    let description = function
        .and_then(|function| function.get("description"))
        .or_else(|| object.get("description"))
        .or_else(|| object.get("server_description"))
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let parameters = function
        .and_then(|function| function.get("parameters"))
        .or_else(|| object.get("parameters"))
        .or_else(|| object.get("input_schema"))
        .cloned()
        .unwrap_or_else(|| default_schema_for_response_tool(kind));
    Some((
        OpenAiTool {
            r#type: "function".to_string(),
            function: OpenAiFunctionTool {
                name,
                description,
                parameters,
            },
        },
        kind.to_string(),
    ))
}

fn default_schema_for_response_tool(kind: &str) -> serde_json::Value {
    match kind {
        "local_shell" => serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}},
                "cmd": {"type": "string"},
                "working_directory": {"type": "string"},
                "timeout_ms": {"type": "integer"}
            }
        }),
        "shell" => serde_json::json!({
            "type": "object",
            "properties": {
                "commands": {"type": "array", "items": {"type": "string"}},
                "cmd": {"type": "string"},
                "timeout_ms": {"type": "integer"},
                "max_output_length": {"type": "integer"}
            }
        }),
        "apply_patch" => serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {"type": "object"},
                "patch": {"type": "string"}
            }
        }),
        "custom" => serde_json::json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }),
        _ => serde_json::json!({"type": "object", "properties": {}}),
    }
}

fn response_tool_choice_to_openai_chat(
    value: &serde_json::Value,
    tools: &mut Vec<OpenAiTool>,
    tool_kinds: &mut BTreeMap<String, String>,
) -> serde_json::Value {
    if value.is_null() {
        return serde_json::Value::Null;
    }
    if value.is_string() {
        return value.clone();
    }
    let Some(object) = value.as_object() else {
        return serde_json::Value::Null;
    };
    let choice_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("function");
    if choice_type == "allowed_tools" {
        return serde_json::Value::String("auto".to_string());
    }
    let name = object
        .get("name")
        .or_else(|| {
            object
                .get("function")
                .and_then(|function| function.get("name"))
        })
        .and_then(|value| value.as_str())
        .unwrap_or(choice_type);
    let name = sanitize_tool_name(name);
    if choice_type == "function" && tools.iter().any(|tool| tool.function.name == name) {
        tools.retain(|tool| tool.function.name == name);
        tool_kinds.retain(|tool_name, _| tool_name == &name);
        return serde_json::Value::String("required".to_string());
    }
    serde_json::json!({
        "type": "function",
        "function": {"name": name}
    })
}

fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .take(64)
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn openai_text_message(role: &str, text: String) -> OpenAiMessage {
    OpenAiMessage {
        role: role.to_string(),
        content: Some(OpenAiContent::Text(text)),
        tool_calls: Vec::new(),
        function_call: None,
        tool_call_id: None,
        name: None,
    }
}

fn format_openai_responses_response(
    execution: &OpenAiResponsesExecution,
    model: &str,
    output: KernelOutput,
    include_public_reasoning: bool,
) -> serde_json::Value {
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));
    let final_text_tool_call = response_text_tool_call_record(execution, &output);
    let output_items = openai_responses_output_items(execution, &output, reasoning_summary);
    let output_text = if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
        && final_text_tool_call.is_none()
    {
        output.final_text.clone()
    } else {
        String::new()
    };
    openai_response_value(
        execution,
        model,
        output_items,
        output_text,
        openai_responses_usage_json(&output.usage),
    )
}

fn response_text_tool_call_record(
    execution: &OpenAiResponsesExecution,
    output: &KernelOutput,
) -> Option<ToolCallRecord> {
    if !output.verification.passed {
        return None;
    }
    let tool_names = execution
        .tool_kinds
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    parse_text_tool_call(&output.final_text, &tool_names)
        .map(parsed_text_tool_call_record)
        .map(|call| normalize_tool_call_for_response_execution(execution, call))
}

fn openai_responses_output_items(
    execution: &OpenAiResponsesExecution,
    output: &KernelOutput,
    reasoning_summary: Option<String>,
) -> Vec<serde_json::Value> {
    let mut items = Vec::new();
    if let Some(summary) = reasoning_summary.filter(|summary| !summary.trim().is_empty()) {
        items.push(serde_json::json!({
            "id": format!("rs_{}", Uuid::new_v4()),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": summary}],
            "status": "completed"
        }));
    }

    if let Some(call) = response_text_tool_call_record(execution, output) {
        items.push(openai_response_tool_call_item(execution, &call));
    } else if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
    {
        items.push(openai_response_message_item(
            &format!("msg_{}", Uuid::new_v4()),
            &output.final_text,
            "completed",
        ));
    } else {
        items.extend(
            output
                .tool_calls
                .iter()
                .cloned()
                .map(|call| normalize_tool_call_for_response_execution(execution, call))
                .map(|call| openai_response_tool_call_item(execution, &call)),
        );
    }
    items
}

fn openai_response_value(
    execution: &OpenAiResponsesExecution,
    model: &str,
    output: Vec<serde_json::Value>,
    output_text: String,
    usage: serde_json::Value,
) -> serde_json::Value {
    openai_response_value_with_status(
        execution,
        model,
        output,
        output_text,
        usage,
        "completed",
        None,
    )
}

fn openai_response_value_with_status(
    execution: &OpenAiResponsesExecution,
    model: &str,
    output: Vec<serde_json::Value>,
    output_text: String,
    usage: serde_json::Value,
    status: &str,
    incomplete_reason: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": execution.response_id,
        "object": "response",
        "created_at": execution.created_at,
        "status": status,
        "background": execution.background,
        "error": null,
        "incomplete_details": incomplete_reason.map(|reason| serde_json::json!({"reason": reason})).unwrap_or(serde_json::Value::Null),
        "instructions": if execution.instructions.is_null() { serde_json::Value::Null } else { execution.instructions.clone() },
        "metadata": execution.metadata,
        "model": model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": execution.parallel_tool_calls,
        "previous_response_id": execution.previous_response_id,
        "store": execution.store,
        "temperature": execution.chat_request.extra.get("temperature").cloned().unwrap_or(serde_json::Value::Null),
        "tool_choice": if execution.tool_choice.is_null() { serde_json::Value::String("auto".to_string()) } else { execution.tool_choice.clone() },
        "tools": execution.tools,
        "top_p": execution.chat_request.extra.get("top_p").cloned().unwrap_or(serde_json::Value::Null),
        "truncation": "disabled",
        "usage": usage
    })
}

fn openai_response_message_item(id: &str, text: &str, status: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": []
        }]
    })
}

fn openai_response_tool_call_item(
    execution: &OpenAiResponsesExecution,
    call: &ToolCallRecord,
) -> serde_json::Value {
    let kind = execution
        .tool_kinds
        .get(&call.tool_name)
        .map(String::as_str)
        .unwrap_or("function");
    match kind {
        "local_shell" => serde_json::json!({
            "id": format!("ls_{}", call.tool_call_id.as_ref()),
            "type": "local_shell_call",
            "call_id": call.tool_call_id.as_ref(),
            "status": "completed",
            "action": local_shell_action_from_arguments(&call.arguments_json)
        }),
        "shell" => serde_json::json!({
            "id": format!("sh_{}", call.tool_call_id.as_ref()),
            "type": "shell_call",
            "call_id": call.tool_call_id.as_ref(),
            "status": "completed",
            "action": shell_action_from_arguments(&call.arguments_json)
        }),
        "apply_patch" => serde_json::json!({
            "id": format!("ap_{}", call.tool_call_id.as_ref()),
            "type": "apply_patch_call",
            "call_id": call.tool_call_id.as_ref(),
            "status": "completed",
            "operation": apply_patch_operation_from_arguments(&call.arguments_json)
        }),
        "custom" => serde_json::json!({
            "id": format!("ct_{}", call.tool_call_id.as_ref()),
            "type": "custom_tool_call",
            "call_id": call.tool_call_id.as_ref(),
            "name": call.tool_name,
            "input": custom_tool_input_from_arguments(&call.arguments_json),
            "status": "completed"
        }),
        _ => serde_json::json!({
            "id": format!("fc_{}", call.tool_call_id.as_ref()),
            "type": "function_call",
            "call_id": call.tool_call_id.as_ref(),
            "name": call.tool_name,
            "arguments": call.arguments_json.to_string(),
            "status": "completed"
        }),
    }
}

fn local_shell_action_from_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    if let Some(action) = arguments.get("action") {
        return action.clone();
    }
    let command = command_array_from_arguments(arguments);
    serde_json::json!({
        "type": "exec",
        "command": command,
        "env": arguments.get("env").cloned().unwrap_or_else(|| serde_json::json!({})),
        "working_directory": arguments.get("working_directory").cloned().unwrap_or(serde_json::Value::Null),
        "timeout_ms": arguments.get("timeout_ms").cloned().unwrap_or(serde_json::Value::Null)
    })
}

fn shell_action_from_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    if let Some(action) = arguments.get("action") {
        return action.clone();
    }
    serde_json::json!({
        "commands": command_array_from_arguments(arguments),
        "timeout_ms": arguments.get("timeout_ms").cloned().unwrap_or(serde_json::Value::Null),
        "max_output_length": arguments.get("max_output_length").cloned().unwrap_or(serde_json::Value::Null)
    })
}

fn command_array_from_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    if let Some(command) = arguments.get("command").and_then(|value| value.as_array()) {
        return serde_json::Value::Array(command.clone());
    }
    if let Some(commands) = arguments.get("commands").and_then(|value| value.as_array()) {
        return serde_json::Value::Array(commands.clone());
    }
    if let Some(cmd) = arguments.get("cmd").and_then(|value| value.as_str()) {
        return serde_json::json!(["sh", "-lc", cmd]);
    }
    if let Some(command) = arguments.get("command").and_then(|value| value.as_str()) {
        return serde_json::json!(["sh", "-lc", command]);
    }
    serde_json::json!([])
}

fn apply_patch_operation_from_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    arguments
        .get("operation")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({
            "type": "update_file",
            "path": arguments.get("path").and_then(|value| value.as_str()).unwrap_or(""),
            "diff": arguments.get("patch").or_else(|| arguments.get("diff")).and_then(|value| value.as_str()).unwrap_or("")
        }))
}

fn custom_tool_input_from_arguments(arguments: &serde_json::Value) -> String {
    arguments
        .get("input")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| arguments.to_string())
}

fn openai_responses_usage_json(usage: &ProviderUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens)
    })
}

async fn maybe_store_openai_response(
    store: &ResponsesStore,
    execution: &OpenAiResponsesExecution,
    response: &serde_json::Value,
    conversation_messages: Vec<OpenAiMessage>,
) -> Result<(), String> {
    store_openai_response(store, execution, response, conversation_messages, false).await
}

async fn store_openai_response(
    store: &ResponsesStore,
    execution: &OpenAiResponsesExecution,
    response: &serde_json::Value,
    conversation_messages: Vec<OpenAiMessage>,
    force: bool,
) -> Result<(), String> {
    if !force && !execution.store {
        return Ok(());
    }
    store
        .put(StoredOpenAiResponse {
            tenant_id: execution.tenant_id.clone(),
            id: execution.response_id.clone(),
            created_at: execution.created_at,
            response: response.clone(),
            conversation_messages,
            input_items: execution.input_items.clone(),
        })
        .await
}

fn response_conversation_messages_from_kernel_output(
    execution: &OpenAiResponsesExecution,
    mut messages: Vec<OpenAiMessage>,
    output: &KernelOutput,
) -> Vec<OpenAiMessage> {
    if let Some(call) = response_text_tool_call_record(execution, output) {
        messages.push(OpenAiMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: vec![openai_message_tool_call_from_record(&call)],
            function_call: None,
            tool_call_id: None,
            name: None,
        });
    } else if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
    {
        messages.push(openai_text_message("assistant", output.final_text.clone()));
    } else if !output.tool_calls.is_empty() {
        messages.push(OpenAiMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: output
                .tool_calls
                .iter()
                .cloned()
                .map(|call| normalize_tool_call_for_response_execution(execution, call))
                .map(|call| openai_message_tool_call_from_record(&call))
                .collect(),
            function_call: None,
            tool_call_id: None,
            name: None,
        });
    }
    messages
}

fn response_conversation_messages_from_chat_completion(
    execution: &OpenAiResponsesExecution,
    mut messages: Vec<OpenAiMessage>,
    value: &serde_json::Value,
) -> Vec<OpenAiMessage> {
    let Some(message) = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
    else {
        return messages;
    };
    if let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
        messages.push(OpenAiMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: tool_calls
                .iter()
                .filter_map(|call| openai_message_tool_call_from_json(execution, call))
                .collect(),
            function_call: None,
            tool_call_id: None,
            name: None,
        });
    } else if let Some(content) = message.get("content").and_then(|value| value.as_str()) {
        let tool_names = execution
            .tool_kinds
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if let Some(call) = parse_text_tool_call(content, &tool_names) {
            messages.push(OpenAiMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: vec![openai_message_tool_call_from_record(
                    &normalize_tool_call_for_response_execution(
                        execution,
                        parsed_text_tool_call_record(call),
                    ),
                )],
                function_call: None,
                tool_call_id: None,
                name: None,
            });
        } else {
            messages.push(openai_text_message("assistant", content.to_string()));
        }
    }
    messages
}

fn openai_message_tool_call_from_json(
    execution: &OpenAiResponsesExecution,
    value: &serde_json::Value,
) -> Option<OpenAiMessageToolCall> {
    let call = tool_call_record_from_chat_json(execution, value)?;
    Some(openai_message_tool_call_from_record(
        &normalize_tool_call_for_response_execution(execution, call),
    ))
}

fn openai_message_tool_call_from_record(call: &ToolCallRecord) -> OpenAiMessageToolCall {
    OpenAiMessageToolCall {
        id: call.tool_call_id.as_ref().to_string(),
        r#type: "function".to_string(),
        function: OpenAiMessageToolCallFunction {
            name: call.tool_name.clone(),
            arguments: call.arguments_json.clone(),
        },
    }
}

fn openai_responses_response_from_chat_completion(
    execution: &OpenAiResponsesExecution,
    value: serde_json::Value,
) -> serde_json::Value {
    let (output, output_text) = chat_completion_response_output_items(execution, &value);
    let usage = value
        .get("usage")
        .and_then(|usage| serde_json::from_value::<ProviderUsage>(usage.clone()).ok())
        .unwrap_or_default();
    let (status, incomplete_reason) = openai_responses_status_from_chat_completion(&value);
    openai_response_value_with_status(
        execution,
        &execution.request_model,
        output,
        output_text,
        openai_responses_usage_json(&usage),
        status,
        incomplete_reason,
    )
}

fn openai_responses_status_from_chat_completion(
    value: &serde_json::Value,
) -> (&'static str, Option<&'static str>) {
    match value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|reason| reason.as_str())
    {
        Some("length") => ("incomplete", Some("max_output_tokens")),
        Some("content_filter") => ("incomplete", Some("content_filter")),
        _ => ("completed", None),
    }
}

fn chat_completion_response_output_items(
    execution: &OpenAiResponsesExecution,
    value: &serde_json::Value,
) -> (Vec<serde_json::Value>, String) {
    let Some(message) = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
    else {
        return (Vec::new(), String::new());
    };
    if let Some(tool_calls) = message
        .get("tool_calls")
        .and_then(|value| value.as_array())
        .filter(|tool_calls| !tool_calls.is_empty())
    {
        let items = tool_calls
            .iter()
            .filter_map(|call| tool_call_record_from_chat_json(execution, call))
            .map(|call| normalize_tool_call_for_response_execution(execution, call))
            .map(|call| openai_response_tool_call_item(execution, &call))
            .collect::<Vec<_>>();
        return (items, String::new());
    }
    let text = message
        .get("content")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let tool_names = execution
        .tool_kinds
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if let Some(call) = parse_text_tool_call(&text, &tool_names) {
        let call = normalize_tool_call_for_response_execution(
            execution,
            parsed_text_tool_call_record(call),
        );
        return (
            vec![openai_response_tool_call_item(execution, &call)],
            String::new(),
        );
    }
    (
        vec![openai_response_message_item(
            &format!("msg_{}", Uuid::new_v4()),
            &text,
            "completed",
        )],
        text,
    )
}

fn tool_call_record_from_chat_json(
    execution: &OpenAiResponsesExecution,
    value: &serde_json::Value,
) -> Option<ToolCallRecord> {
    let function = value.get("function")?;
    let id = value
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or("call-unknown");
    let name = function.get("name")?.as_str()?.to_string();
    let arguments = function
        .get("arguments")
        .cloned()
        .map(openai_tool_arguments)
        .unwrap_or(serde_json::Value::Null);
    Some(ToolCallRecord {
        tool_call_id: ToolCallId::from(id),
        scope: IsolationKey::new(
            &execution.tenant_id,
            &execution.response_id,
            &execution.response_id,
        ),
        task_id: TaskId::from("root"),
        agent_id: AgentId::from("agent-root"),
        tool_name: name,
        arguments_sha256: sha256_hex(&arguments.to_string()),
        arguments_json: arguments,
        status: ToolCallStatus::Pending,
        created_at_ms: execution.created_at.saturating_mul(1000),
        resolved_at_ms: None,
    })
}

fn normalize_tool_call_for_response_execution(
    execution: &OpenAiResponsesExecution,
    call: ToolCallRecord,
) -> ToolCallRecord {
    if execution.tool_kinds.contains_key(&call.tool_name) {
        return call;
    }
    let Some((tool_name, arguments_json)) =
        adapt_unknown_tool_call_to_available_tool(execution, &call.tool_name, &call.arguments_json)
    else {
        return call;
    };

    ToolCallRecord {
        tool_name,
        arguments_sha256: sha256_hex(&arguments_json.to_string()),
        arguments_json,
        ..call
    }
}

fn adapt_unknown_tool_call_to_available_tool(
    execution: &OpenAiResponsesExecution,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Option<(String, serde_json::Value)> {
    let exec_tool_name = available_shell_exec_tool_name(execution)?;
    let canonical_name = canonical_tool_alias(tool_name);
    let command = match canonical_name.as_str() {
        "readfile" | "readtextfile" | "openfile" | "getfilecontents" => {
            let path = argument_string(arguments, &["path", "file_path", "filepath", "filename"])?;
            format!("cat -- {}", shell_quote(&path))
        }
        "listfiles" | "listdirectory" | "listdir" => {
            let path = argument_string(arguments, &["path", "directory", "dir"])
                .unwrap_or_else(|| ".".to_string());
            format!("ls -la -- {}", shell_quote(&path))
        }
        "searchfiles" | "searchfilecontents" | "grepfiles" => {
            let pattern = argument_string(arguments, &["pattern", "query", "regex"])?;
            let path = argument_string(arguments, &["path", "directory", "dir"])
                .unwrap_or_else(|| ".".to_string());
            format!(
                "rg --line-number --hidden --glob {} -- {} {}",
                shell_quote("!.git"),
                shell_quote(&pattern),
                shell_quote(&path)
            )
        }
        _ => return None,
    };

    Some((exec_tool_name, serde_json::json!({ "cmd": command })))
}

fn available_shell_exec_tool_name(execution: &OpenAiResponsesExecution) -> Option<String> {
    ["exec_command", "local_shell", "shell"]
        .iter()
        .find(|name| execution.tool_kinds.contains_key(**name))
        .map(|name| (*name).to_string())
}

fn canonical_tool_alias(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn argument_string(arguments: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        arguments
            .get(*key)
            .and_then(|value| value.as_str())
            .map(str::to_string)
    })
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn openai_chat_request_json(request: &OpenAiChatRequest, stream: bool) -> serde_json::Value {
    let mut value = serde_json::to_value(request).unwrap_or_else(|_| serde_json::json!({}));
    value["stream"] = serde_json::Value::Bool(stream);
    if value
        .get("tools")
        .and_then(|tools| tools.as_array())
        .is_some_and(Vec::is_empty)
    {
        value.as_object_mut().map(|object| object.remove("tools"));
    }
    value
}

fn format_openai_responses_stream_response_from_value(response: serde_json::Value) -> Response {
    let mut body = String::new();
    body.push_str(&sse_event(
        "response.created",
        serde_json::json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": response_with_status(response.clone(), "in_progress")
        }),
    ));
    body.push_str(&openai_responses_output_events(&response, 1));
    body.push_str(&sse_event(
        "response.completed",
        serde_json::json!({
            "type": "response.completed",
            "sequence_number": 10_000,
            "response": response
        }),
    ));
    body.push_str("data: [DONE]\n\n");
    sse_response(body)
}

fn openai_responses_output_events(response: &serde_json::Value, start_seq: u64) -> String {
    let mut body = String::new();
    for event in openai_responses_output_event_values(response, start_seq) {
        let event_name = event
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("response.event")
            .to_string();
        body.push_str(&sse_event(&event_name, event));
    }
    body
}

fn openai_responses_output_event_values(
    response: &serde_json::Value,
    start_seq: u64,
) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    let mut sequence_number = start_seq;
    let output = response
        .get("output")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    for (output_index, item) in output.into_iter().enumerate() {
        events.push(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": response_item_with_status(item.clone(), "in_progress")
        }));
        sequence_number += 1;
        match item.get("type").and_then(|value| value.as_str()) {
            Some("message") => {
                let item_id = item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("msg");
                let text = item
                    .get("content")
                    .and_then(|value| value.as_array())
                    .and_then(|content| content.first())
                    .and_then(|part| part.get("text"))
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let part = serde_json::json!({
                    "type": "output_text",
                    "text": "",
                    "annotations": []
                });
                events.push(serde_json::json!({
                    "type": "response.content_part.added",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part
                }));
                sequence_number += 1;
                if !text.is_empty() {
                    events.push(serde_json::json!({
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": []
                    }));
                    sequence_number += 1;
                }
                let done_part = serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                });
                events.push(serde_json::json!({
                    "type": "response.output_text.done",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": []
                }));
                sequence_number += 1;
                events.push(serde_json::json!({
                    "type": "response.content_part.done",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": done_part
                }));
                sequence_number += 1;
            }
            Some("function_call") => {
                let item_id = item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .unwrap_or("fc");
                let arguments = item
                    .get("arguments")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let name = item
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                if !arguments.is_empty() {
                    events.push(serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence_number,
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": arguments
                    }));
                    sequence_number += 1;
                }
                events.push(serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": output_index,
                    "name": name,
                    "arguments": arguments
                }));
                sequence_number += 1;
            }
            _ => {}
        }
        events.push(serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": item
        }));
        sequence_number += 1;
    }
    events
}

fn response_with_status(mut response: serde_json::Value, status: &str) -> serde_json::Value {
    response["status"] = serde_json::Value::String(status.to_string());
    response
}

fn response_item_with_status(mut item: serde_json::Value, status: &str) -> serde_json::Value {
    if item.get("status").is_some() {
        item["status"] = serde_json::Value::String(status.to_string());
    }
    item
}

fn format_openai_responses_provider_stream_response(
    execution: OpenAiResponsesExecution,
    provider_stream: ProviderStream,
    store: ResponsesStore,
    telemetry_context: TelemetryContext,
) -> Response {
    let state = OpenAiResponsesStreamState {
        execution,
        provider_stream,
        store,
        telemetry_context,
        usage: ProviderUsage::default(),
        pending: VecDeque::new(),
        output_text: String::new(),
        text_item_id: None,
        tool_calls: BTreeMap::new(),
        sequence_number: 0,
        output_started: false,
        saw_tool_delta: false,
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
                Some(Ok(event)) => state.push_provider_event(event).await,
                Some(Err(error)) => {
                    let sequence_number = state.next_sequence();
                    let failed_response = state.current_response("failed");
                    state.pending.push_back(sse_event(
                        "response.failed",
                        serde_json::json!({
                            "type": "response.failed",
                            "sequence_number": sequence_number,
                            "response": failed_response
                        }),
                    ));
                    state.pending.push_back(sse_event(
                        "error",
                        serde_json::json!({
                            "type": "error",
                            "message": error.to_string()
                        }),
                    ));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
                    state.emit_stream_telemetry(None, Some("provider_stream_error"));
                }
                None => state.finish(ProviderFinishReason::Stop).await,
            }
        }
    }))
}

#[derive(Clone)]
struct PendingResponseToolCall {
    index: usize,
    id: ToolCallId,
    name: String,
    arguments: String,
}

struct OpenAiResponsesStreamState {
    execution: OpenAiResponsesExecution,
    provider_stream: ProviderStream,
    store: ResponsesStore,
    telemetry_context: TelemetryContext,
    usage: ProviderUsage,
    pending: VecDeque<String>,
    output_text: String,
    text_item_id: Option<String>,
    tool_calls: BTreeMap<usize, PendingResponseToolCall>,
    sequence_number: u64,
    output_started: bool,
    saw_tool_delta: bool,
    finished: bool,
    telemetry_emitted: bool,
}

impl OpenAiResponsesStreamState {
    async fn push_provider_event(&mut self, event: ProviderStreamEvent) {
        match event {
            ProviderStreamEvent::TextDelta { text } if !text.is_empty() => {
                self.ensure_created();
                self.ensure_text_output();
                let item_id = self.text_item_id.clone().unwrap_or_default();
                self.output_text.push_str(&text);
                let seq = self.next_sequence();
                self.pending.push_back(sse_event(
                    "response.output_text.delta",
                    serde_json::json!({
                        "type": "response.output_text.delta",
                        "sequence_number": seq,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": []
                    }),
                ));
            }
            ProviderStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                self.ensure_created();
                self.saw_tool_delta = true;
                let mut added_item = None;
                let mut delta_item_id = None;
                {
                    let call = self.tool_calls.entry(index).or_insert_with(|| {
                        let id = id
                            .clone()
                            .unwrap_or_else(|| ToolCallId::from(format!("call-{index}")));
                        PendingResponseToolCall {
                            index,
                            id,
                            name: name.clone().unwrap_or_else(|| "unknown".to_string()),
                            arguments: String::new(),
                        }
                    });
                    if let Some(name) = name {
                        call.name = name;
                    }
                    let was_empty = call.arguments.is_empty();
                    call.arguments.push_str(&arguments_delta);
                    if was_empty {
                        added_item = Some(pending_response_tool_call_item(&self.execution, call));
                    }
                    if !arguments_delta.is_empty() {
                        delta_item_id = Some(response_tool_item_id(&self.execution, call));
                    }
                }
                if let Some(item) = added_item {
                    let seq = self.next_sequence();
                    self.pending.push_back(sse_event(
                        "response.output_item.added",
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": seq,
                            "output_index": index,
                            "item": response_item_with_status(item, "in_progress")
                        }),
                    ));
                }
                if let Some(item_id) = delta_item_id {
                    let seq = self.next_sequence();
                    self.pending.push_back(sse_event(
                        "response.function_call_arguments.delta",
                        serde_json::json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": seq,
                            "item_id": item_id,
                            "output_index": index,
                            "delta": arguments_delta
                        }),
                    ));
                }
            }
            ProviderStreamEvent::Finish { reason } => self.finish(reason).await,
            ProviderStreamEvent::Usage { usage } => {
                self.usage = usage;
            }
            ProviderStreamEvent::TextDelta { .. } => {}
        }
    }

    fn ensure_created(&mut self) {
        if self.output_started {
            return;
        }
        self.output_started = true;
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.created",
            serde_json::json!({
                "type": "response.created",
                "sequence_number": seq,
                "response": self.current_response("in_progress")
            }),
        ));
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.in_progress",
            serde_json::json!({
                "type": "response.in_progress",
                "sequence_number": seq,
                "response": self.current_response("in_progress")
            }),
        ));
    }

    fn ensure_text_output(&mut self) {
        if self.text_item_id.is_some() {
            return;
        }
        let item_id = format!("msg_{}", Uuid::new_v4());
        self.text_item_id = Some(item_id.clone());
        let item = openai_response_message_item(&item_id, "", "in_progress");
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.output_item.added",
            serde_json::json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": 0,
                "item": item
            }),
        ));
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.content_part.added",
            serde_json::json!({
                "type": "response.content_part.added",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}
            }),
        ));
    }

    async fn finish(&mut self, reason: ProviderFinishReason) {
        self.ensure_created();
        if self.saw_tool_delta {
            self.finish_tool_calls();
        } else {
            self.finish_text();
        }
        let response = self.current_response("completed");
        let conversation_messages = if self.saw_tool_delta {
            response_conversation_messages_from_pending_tool_calls(
                &self.execution,
                self.execution.conversation_messages.clone(),
                self.tool_calls.values().cloned().collect(),
            )
        } else {
            let mut messages = self.execution.conversation_messages.clone();
            messages.push(openai_text_message("assistant", self.output_text.clone()));
            messages
        };
        if let Err(error) = maybe_store_openai_response(
            &self.store,
            &self.execution,
            &response,
            conversation_messages,
        )
        .await
        {
            self.pending.push_back(sse_event(
                "error",
                serde_json::json!({
                    "type": "error",
                    "message": error
                }),
            ));
        }
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.completed",
            serde_json::json!({
                "type": "response.completed",
                "sequence_number": seq,
                "response": response
            }),
        ));
        self.pending.push_back("data: [DONE]\n\n".to_string());
        self.finished = true;
        self.emit_stream_telemetry(Some(openai_responses_finish_reason(&reason)), None);
    }

    fn finish_text(&mut self) {
        self.ensure_text_output();
        let item_id = self.text_item_id.clone().unwrap_or_default();
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.output_text.done",
            serde_json::json!({
                "type": "response.output_text.done",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": self.output_text,
                "logprobs": []
            }),
        ));
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.content_part.done",
            serde_json::json!({
                "type": "response.content_part.done",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.output_text, "annotations": []}
            }),
        ));
        let seq = self.next_sequence();
        self.pending.push_back(sse_event(
            "response.output_item.done",
            serde_json::json!({
                "type": "response.output_item.done",
                "sequence_number": seq,
                "output_index": 0,
                "item": openai_response_message_item(&item_id, &self.output_text, "completed")
            }),
        ));
    }

    fn finish_tool_calls(&mut self) {
        let calls = self.tool_calls.values().cloned().collect::<Vec<_>>();
        for call in calls {
            let record = pending_response_tool_call_record(&self.execution, &call);
            let item_id = response_tool_item_id(&self.execution, &call);
            let seq = self.next_sequence();
            self.pending.push_back(sse_event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": seq,
                    "item_id": item_id,
                    "output_index": call.index,
                    "name": record.tool_name,
                    "arguments": record.arguments_json.to_string()
                }),
            ));
            let seq = self.next_sequence();
            self.pending.push_back(sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": seq,
                    "output_index": call.index,
                    "item": pending_response_tool_call_item(&self.execution, &call)
                }),
            ));
        }
    }

    fn current_response(&self, status: &str) -> serde_json::Value {
        let output = if self.saw_tool_delta {
            self.tool_calls
                .values()
                .map(|call| pending_response_tool_call_item(&self.execution, call))
                .collect::<Vec<_>>()
        } else if let Some(item_id) = &self.text_item_id {
            vec![openai_response_message_item(
                item_id,
                &self.output_text,
                if status == "completed" {
                    "completed"
                } else {
                    "in_progress"
                },
            )]
        } else {
            Vec::new()
        };
        let mut response = openai_response_value(
            &self.execution,
            &self.execution.request_model,
            output,
            self.output_text.clone(),
            openai_responses_usage_json(&self.usage),
        );
        response["status"] = serde_json::Value::String(status.to_string());
        response
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence_number;
        self.sequence_number = self.sequence_number.saturating_add(1);
        sequence
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

fn pending_response_tool_call_item(
    execution: &OpenAiResponsesExecution,
    call: &PendingResponseToolCall,
) -> serde_json::Value {
    let record = pending_response_tool_call_record(execution, call);
    openai_response_tool_call_item(execution, &record)
}

fn pending_response_tool_call_record(
    execution: &OpenAiResponsesExecution,
    call: &PendingResponseToolCall,
) -> ToolCallRecord {
    let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
    let record = ToolCallRecord {
        tool_call_id: call.id.clone(),
        scope: IsolationKey::new(
            &execution.tenant_id,
            &execution.response_id,
            &execution.response_id,
        ),
        task_id: TaskId::from("root"),
        agent_id: AgentId::from("agent-root"),
        tool_name: call.name.clone(),
        arguments_sha256: sha256_hex(&arguments.to_string()),
        arguments_json: arguments,
        status: ToolCallStatus::Pending,
        created_at_ms: execution.created_at.saturating_mul(1000),
        resolved_at_ms: None,
    };
    normalize_tool_call_for_response_execution(execution, record)
}

fn response_tool_item_id(
    execution: &OpenAiResponsesExecution,
    call: &PendingResponseToolCall,
) -> String {
    pending_response_tool_call_item(execution, call)
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or("tool")
        .to_string()
}

fn response_conversation_messages_from_pending_tool_calls(
    execution: &OpenAiResponsesExecution,
    mut messages: Vec<OpenAiMessage>,
    calls: Vec<PendingResponseToolCall>,
) -> Vec<OpenAiMessage> {
    if calls.is_empty() {
        return messages;
    }
    messages.push(OpenAiMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: calls
            .into_iter()
            .map(|call| {
                openai_message_tool_call_from_record(&pending_response_tool_call_record(
                    execution, &call,
                ))
            })
            .collect(),
        function_call: None,
        tool_call_id: None,
        name: None,
    });
    messages
}

fn openai_responses_finish_reason(reason: &ProviderFinishReason) -> &'static str {
    match reason {
        ProviderFinishReason::ToolCalls | ProviderFinishReason::FunctionCall => "tool_calls",
        ProviderFinishReason::Length => "length",
        _ => "stop",
    }
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
    tool_names: &BTreeSet<String>,
    tool_response_format: OpenAiToolResponseFormat,
    include_public_reasoning: bool,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let mut body = String::new();
    body.push_str(&openai_stream_role_chunk(&id, &model));
    let reasoning_summary = include_public_reasoning.then(|| public_reasoning_summary(&output));

    let text_tool_call = output
        .verification
        .passed
        .then(|| parse_text_tool_call(&output.final_text, tool_names))
        .flatten()
        .map(parsed_text_tool_call_record);

    if let Some(call) = text_tool_call {
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
        if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
            body.push_str(&sse_data(openai_stream_chunk(
                &id,
                &model,
                serde_json::json!({"function_call": openai_legacy_function_call_json(call)}),
                serde_json::Value::Null,
            )));
            body.push_str(&openai_stream_finish_chunk(&id, &model, "function_call"));
        } else {
            body.push_str(&sse_data(openai_stream_chunk(
                &id,
                &model,
                serde_json::json!({"tool_calls": [openai_stream_tool_call_delta(0, call)]}),
                serde_json::Value::Null,
            )));
            body.push_str(&openai_stream_finish_chunk(&id, &model, "tool_calls"));
        }
    } else if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
    {
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

fn openai_responses_public_reasoning_requested(request: &OpenAiResponsesRequest) -> bool {
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
    let stop_reason = if output.verification.unresolved_tool_calls.is_empty()
        && !output.final_text.trim().is_empty()
    {
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

    if output.verification.unresolved_tool_calls.is_empty() && !output.final_text.trim().is_empty()
    {
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

fn parse_openai_responses_request(
    raw: serde_json::Value,
) -> Result<OpenAiResponsesRequest, ApiError> {
    serde_json::from_value(raw).map_err(|error| {
        ApiError::InvalidRequest(format!("invalid OpenAI Responses request: {error}"))
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
                        OpenAiContentPart::InputAudio { input_audio, extra } => {
                            content.push(NormalizedContentPart::ProviderContent {
                                source_format: SourceFormat::OpenAIChat,
                                value: tagged_provider_content(
                                    "input_audio",
                                    "input_audio",
                                    input_audio,
                                    extra,
                                ),
                            });
                        }
                        OpenAiContentPart::File { file, extra } => {
                            content.push(NormalizedContentPart::ProviderContent {
                                source_format: SourceFormat::OpenAIChat,
                                value: tagged_provider_content("file", "file", file, extra),
                            });
                        }
                        OpenAiContentPart::Refusal { refusal } => {
                            content.push(NormalizedContentPart::ProviderContent {
                                source_format: SourceFormat::OpenAIChat,
                                value: serde_json::json!({"type": "refusal", "refusal": refusal}),
                            });
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
                        AnthropicContentBlock::ProviderContent { value } => {
                            content.push(NormalizedContentPart::ProviderContent {
                                source_format: SourceFormat::AnthropicMessages,
                                value,
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
            | "reasoning_effort"
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

    if let Some(effort) = top_level_reasoning_effort(&request.extra)? {
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

    if let Some(effort) = top_level_reasoning_effort(&request.extra)? {
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

fn top_level_reasoning_effort(
    extra: &BTreeMap<String, serde_json::Value>,
) -> Result<Option<ReasoningEffort>, ApiError> {
    extra
        .get("reasoning_effort")
        .and_then(|value| value.as_str())
        .map(parse_reasoning_effort)
        .transpose()
}

fn reasoning_with_top_level_effort(
    reasoning: &serde_json::Value,
    extra: &BTreeMap<String, serde_json::Value>,
) -> serde_json::Value {
    let Some(effort) = extra
        .get("reasoning_effort")
        .and_then(|value| value.as_str())
    else {
        return reasoning.clone();
    };

    let mut object = reasoning.as_object().cloned().unwrap_or_default();
    object.insert(
        "effort".to_string(),
        serde_json::Value::String(effort.to_string()),
    );
    serde_json::Value::Object(object)
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
        "system" | "developer" => MessageRole::System,
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
                AnthropicContentBlock::ProviderContent { value } => {
                    Some(NormalizedContentPart::ProviderContent {
                        source_format: SourceFormat::AnthropicMessages,
                        value,
                    })
                }
            })
            .collect(),
    }
}

fn tagged_provider_content(
    kind: &str,
    field: &str,
    payload: serde_json::Value,
    extra: BTreeMap<String, serde_json::Value>,
) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "type".to_string(),
        serde_json::Value::String(kind.to_string()),
    );
    object.insert(field.to_string(), payload);
    object.extend(extra);
    serde_json::Value::Object(object)
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

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn json_bool(value: serde_json::Value) -> Option<bool> {
    value.as_bool().or_else(|| {
        value.as_str().map(|text| {
            matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "true" | "1" | "yes" | "on"
            )
        })
    })
}

fn estimate_response_input_tokens(input: &serde_json::Value) -> u32 {
    estimate_text_tokens(&response_input_text_for_compaction(input))
}

fn response_input_text_for_compaction(input: &serde_json::Value) -> String {
    if let Some(text) = input.as_str() {
        return text.to_string();
    }
    if let Some(items) = input.as_array() {
        return items
            .iter()
            .map(response_input_text_for_compaction)
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    if let Some(object) = input.as_object() {
        if let Some(content) = object.get("content") {
            return response_input_text_for_compaction(content);
        }
        if let Some(text) = object.get("text").and_then(|value| value.as_str()) {
            return text.to_string();
        }
        if let Some(output) = object.get("output") {
            return output
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| output.to_string());
        }
    }
    String::new()
}

fn estimate_text_tokens(text: &str) -> u32 {
    if text.trim().is_empty() {
        return 0;
    }

    let mut tokens = 0_u64;
    let mut ascii_run = 0_u64;
    let flush_ascii_run = |tokens: &mut u64, ascii_run: &mut u64| {
        if *ascii_run > 0 {
            *tokens = tokens.saturating_add(ascii_run.saturating_add(3) / 4);
            *ascii_run = 0;
        }
    };

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            ascii_run = ascii_run.saturating_add(1);
            continue;
        }

        flush_ascii_run(&mut tokens, &mut ascii_run);
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_ascii() {
            tokens = tokens.saturating_add(1);
        } else {
            tokens = tokens.saturating_add((ch.len_utf8() as u64).saturating_add(2) / 3);
        }
    }
    flush_ascii_run(&mut tokens, &mut ascii_run);

    tokens.max(1).min(u64::from(u32::MAX)) as u32
}

pub fn crate_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{MediaSource, NormalizedContentPart, SourceFormat};

    #[test]
    fn local_token_estimate_handles_cjk_and_long_unspaced_text() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert!(estimate_text_tokens("這是一段沒有空格的中文內容") >= 12);
        assert_eq!(estimate_text_tokens("abcdefgh"), 2);
        assert!(estimate_text_tokens("hello, world!") >= 4);
    }

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
    fn normalization_openai_preserves_audio_and_file_content_blocks() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                    {"type": "file", "file": {"file_id": "file-123", "filename": "facts.pdf"}}
                ]
            }]
        }))
        .unwrap();
        let normalized = normalize_openai_chat(request).unwrap();
        assert!(matches!(
            &normalized.messages[0].content[0],
            NormalizedContentPart::ProviderContent { source_format: SourceFormat::OpenAIChat, value }
                if value["type"] == "input_audio"
        ));
        assert!(matches!(
            &normalized.messages[0].content[1],
            NormalizedContentPart::ProviderContent { source_format: SourceFormat::OpenAIChat, value }
                if value["file"]["file_id"] == "file-123"
        ));
    }

    #[test]
    fn normalization_openai_developer_role_preserves_instruction_priority() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "messages": [
                {"role": "developer", "content": "Return JSON only."},
                {"role": "user", "content": "Give me an answer."}
            ]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.messages[0].role, MessageRole::System);
        assert_eq!(normalized.messages[1].role, MessageRole::User);
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
    fn normalization_anthropic_preserves_document_citation_and_server_tool_blocks() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "facts"}, "title": "facts"},
                    {"type": "text", "text": "cited text", "citations": [{"type": "char_location", "cited_text": "cited", "document_index": 0, "document_title": "facts", "start_char_index": 0, "end_char_index": 5}]},
                    {"type": "server_tool_use", "id": "srv-1", "name": "web_search", "input": {"query": "facts"}}
                ]
            }]
        }))
        .unwrap();
        let normalized = normalize_anthropic_messages(request).unwrap();
        let values = normalized.messages[0]
            .content
            .iter()
            .filter_map(|part| match part {
                NormalizedContentPart::ProviderContent { value, .. } => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0]["type"], "document");
        assert!(values[1]["citations"].is_array());
        assert_eq!(values[2]["type"], "server_tool_use");
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
    fn normalization_openai_top_level_reasoning_effort_controls_orchestration_policy() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
            "messages": [{"role": "user", "content": "use top-level effort"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::High);
        assert_eq!(normalized.reasoning_effort.max_agents(), 32);
        assert!(
            normalized
                .provider_options
                .get("reasoning_effort")
                .is_none()
        );
        assert_eq!(normalized.provider_options["reasoning"]["summary"], "auto");
    }

    #[test]
    fn normalization_openai_model_alias_overrides_top_level_reasoning_effort() {
        let request: OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "custom-model-xhigh",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "model suffix wins"}]
        }))
        .unwrap();

        let normalized = normalize_openai_chat(request).unwrap();

        assert_eq!(normalized.model, "custom-model");
        assert_eq!(normalized.reasoning_effort, ReasoningEffort::XHigh);
        assert_eq!(normalized.reasoning_effort.max_agents(), 64);
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
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
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
        assert!(
            normalized
                .provider_options
                .get("reasoning_effort")
                .is_none()
        );
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
    fn normalization_anthropic_top_level_reasoning_effort_controls_orchestration_policy() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "mock",
            "max_tokens": 256,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "use top-level effort"}]
        }))
        .unwrap();

        let normalized = normalize_anthropic_messages(request).unwrap();

        assert_eq!(normalized.reasoning_effort, ReasoningEffort::Low);
        assert_eq!(normalized.reasoning_effort.max_agents(), 4);
        assert!(
            normalized
                .provider_options
                .get("reasoning_effort")
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
        let limiter =
            TenantConcurrencyLimiter::from_env_values(None, DEFAULT_TENANT_QUEUE_TIMEOUT_MS);

        assert_eq!(
            limiter.max_per_tenant,
            Some(DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS)
        );
        assert_eq!(
            DEFAULT_TENANT_MAX_CONCURRENT_REQUESTS,
            ReasoningEffort::Medium.max_agents() as usize
        );
    }

    #[tokio::test]
    async fn tenant_concurrency_limiter_returns_bounded_overload_instead_of_waiting_forever() {
        let limiter =
            TenantConcurrencyLimiter::with_max_per_tenant_and_wait(1, Duration::from_millis(5));
        let tenant = TenantId::from("slow-tenant");
        let _first = limiter.acquire(&tenant).await.unwrap();

        let error = limiter.acquire(&tenant).await.err().unwrap();

        assert_eq!(error, ProviderError::QueueTimeout { wait_ms: 5 });
    }

    #[tokio::test]
    async fn tenant_concurrency_limiter_prunes_inactive_tenant_semaphores() {
        let limiter = TenantConcurrencyLimiter::with_max_per_tenant(1);
        let first_tenant = TenantId::from("inactive-tenant");
        let first = limiter.acquire(&first_tenant).await.unwrap();
        drop(first);

        let second_tenant = TenantId::from("active-tenant");
        let _second = limiter.acquire(&second_tenant).await.unwrap();

        let semaphores = limiter.semaphores.lock().unwrap();
        assert_eq!(semaphores.len(), 1);
        assert!(semaphores.contains_key("active-tenant"));
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
    fn direct_openai_sanitizer_does_not_invent_qwen_thinking_controls() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "local-qwen-model",
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": "hello"}]
        }));

        assert!(sanitized.get("reasoning").is_none());
        assert!(sanitized.get("chat_template_kwargs").is_none());
        assert!(sanitized.get("enable_thinking").is_none());
        assert!(sanitized.get("thinking").is_none());
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
            "reasoning_effort": "high",
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
        assert!(sanitized.get("reasoning_effort").is_none());
        assert_eq!(sanitized["metadata"]["foo"], "bar");
        assert!(sanitized["metadata"].get("tenant_id").is_none());
        assert!(sanitized["metadata"].get("context").is_none());
    }

    #[test]
    fn direct_openai_sanitizer_adapts_named_tool_choice() {
        let sanitized = sanitize_direct_openai_request(serde_json::json!({
            "model": "local-openai-compatible-model",
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

    #[test]
    fn direct_openai_response_converts_token_tool_call_markup() {
        let request = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {"type": "object"}
                }
            }]
        });
        let response = strip_direct_openai_response_with_tools(
            serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<|tool_call>call:exec_command{cmd:<|\"|>cat /app/data.txt<|\"|>}<tool_call|>"
                    },
                    "finish_reason": "stop"
                }]
            }),
            &openai_request_tool_names(&request),
        );

        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "exec_command"
        );
        let arguments: serde_json::Value = serde_json::from_str(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            arguments["cmd"],
            serde_json::Value::String("cat /app/data.txt".to_string())
        );
        assert_eq!(response["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn direct_openai_response_converts_unknown_text_tool_to_safe_executor_observation() {
        let request = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {"type": "object"}
                }
            }]
        });
        let response = strip_direct_openai_response_with_tools(
            serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<|tool_call>call:update_plan{steps:[{status:<|\"|>in_progress<|\"|>}]}<tool_call|>"
                    },
                    "finish_reason": "stop"
                }]
            }),
            &openai_request_tool_names(&request),
        );

        let arguments: serde_json::Value = serde_json::from_str(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "exec_command"
        );
        assert!(
            arguments["cmd"]
                .as_str()
                .unwrap()
                .contains("Unsupported model-emitted tool")
        );
    }

    #[test]
    fn direct_openai_response_parses_marker_quoted_tool_argument_with_inner_quotes() {
        let request = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {"type": "object"}
                }
            }]
        });
        let response = strip_direct_openai_response_with_tools(
            serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<|tool_call>call:exec_command{cmd:<|\"|>cat /app/decomp.c || echo \"Compilation failed\"<|\"|>}<tool_call|>"
                    },
                    "finish_reason": "stop"
                }]
            }),
            &openai_request_tool_names(&request),
        );

        let arguments: serde_json::Value = serde_json::from_str(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            arguments["cmd"],
            "cat /app/decomp.c || echo \"Compilation failed\""
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
        assert_eq!(value["has_more"], false);
        assert_eq!(value["data"][0]["type"], "model");
        assert_eq!(value["data"][0]["display_name"], "mock");
        assert_eq!(value["data"][0]["created_at"], "1970-01-01T00:00:00Z");
    }

    #[tokio::test]
    async fn routes_models_retrieve_supports_current_sdk_shape() {
        let found = tower::ServiceExt::oneshot(
            build_router(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/models/mock")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(found.status(), axum::http::StatusCode::OK);
        let value = response_json(found).await;
        assert_eq!(value["id"], "mock");
        assert_eq!(value["object"], "model");
        assert_eq!(value["type"], "model");

        let missing = tower::ServiceExt::oneshot(
            build_router(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/models/missing-model")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn routes_anthropic_count_tokens_supports_current_sdk_shape() {
        let response = tower::ServiceExt::oneshot(
            build_router(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "system": "請使用繁體中文回答。",
                        "messages": [{
                            "role": "user",
                            "content": [{"type": "text", "text": "這是一段沒有空格的中文內容"}]
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
        assert!(value["input_tokens"].as_u64().unwrap() > 10);
    }

    #[tokio::test]
    async fn shared_api_key_accepts_openai_and_anthropic_sdk_headers() {
        let state = AppState {
            shared_api_key: Some(Arc::<str>::from("deployment-shared-key")),
            ..AppState::default()
        };
        let app = build_router_with_state(state);

        let missing = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(missing.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert_eq!(
            missing.headers()[axum::http::header::WWW_AUTHENTICATE],
            "Bearer"
        );

        let bearer = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("authorization", "Bearer deployment-shared-key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(bearer.status(), axum::http::StatusCode::OK);

        let anthropic = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", "deployment-shared-key")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "reasoning_effort": "none",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "OK"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(anthropic.status(), axum::http::StatusCode::OK);

        let health = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("GET")
                .uri("/health")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(health.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn shared_api_key_uses_anthropic_authentication_error_shape() {
        let state = AppState {
            shared_api_key: Some(Arc::<str>::from("deployment-shared-key")),
            ..AppState::default()
        };
        let response = tower::ServiceExt::oneshot(
            build_router_with_state(state),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", "wrong-key")
                .body(axum::body::Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
        let value = response_json(response).await;
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "authentication_error");
    }

    #[test]
    fn kernel_timeouts_and_provider_queue_errors_map_to_gateway_statuses() {
        let timeout = kernel_error_response(KernelError::RequestTimeout { timeout_ms: 50 });
        assert_eq!(timeout.status(), axum::http::StatusCode::GATEWAY_TIMEOUT);

        let overloaded = kernel_error_response(KernelError::ProviderRejected(
            "provider queue wait exceeded 100 ms".to_string(),
        ));
        assert_eq!(
            overloaded.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );

        let provider_overloaded =
            provider_error_response(ProviderError::QueueTimeout { wait_ms: 30_000 });
        assert_eq!(
            provider_overloaded.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(provider_overloaded.headers()[header::RETRY_AFTER], "30");
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
    async fn routes_openai_responses_nonstream_returns_response_shape() {
        let app = build_router();
        let value = post_json(
            app,
            "/v1/responses",
            serde_json::json!({
                "model": "mock",
                "input": "hello responses"
            }),
        )
        .await;

        assert_eq!(value["object"], "response");
        assert!(value["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output"][0]["type"], "message");
        assert_eq!(value["output"][0]["role"], "assistant");
        assert_eq!(value["output"][0]["content"][0]["type"], "output_text");
        assert!(
            value["output_text"]
                .as_str()
                .unwrap()
                .contains("clear, usable answer")
        );
        assert_eq!(value["usage"]["total_tokens"], 0);
    }

    #[tokio::test]
    async fn responses_direct_chat_empty_tool_calls_preserves_text_output() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "adapter text",
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let response = openai_responses_response_from_chat_completion(
            &execution,
            serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "adapter text output",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7
                }
            }),
        );

        assert_eq!(response["object"], "response");
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "adapter text output"
        );
        assert_eq!(response["output_text"], "adapter text output");
    }

    #[tokio::test]
    async fn responses_chat_request_omits_null_message_fields_for_strict_backends() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "adapter text",
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let raw = openai_chat_request_json(&execution.chat_request, false);
        let message = &raw["messages"][0];

        assert_eq!(message["role"], "user");
        assert!(message.get("name").is_none());
        assert!(message.get("function_call").is_none());
        assert!(message.get("tool_call_id").is_none());
        assert!(message.get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn responses_direct_chat_length_finish_maps_to_incomplete() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "adapter text",
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let response = openai_responses_response_from_chat_completion(
            &execution,
            serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "private reasoning that did not reach a final answer",
                        "tool_calls": []
                    },
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7
                }
            }),
        );

        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert_eq!(response["output_text"], "");
        assert!(
            !response["output"]
                .to_string()
                .contains("private reasoning that did not reach a final answer")
        );
    }

    #[tokio::test]
    async fn responses_function_tool_without_description_omits_null_description() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "use the tool",
            "tools": [{
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}}
            }],
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let raw = openai_chat_request_json(&execution.chat_request, false);

        assert!(raw["tools"][0]["function"].get("description").is_none());
        assert_eq!(raw["tools"][0]["function"]["name"], "lookup");
        assert!(raw["tools"][0]["function"]["parameters"].is_object());
    }

    #[tokio::test]
    async fn responses_named_tool_choice_filters_to_required_tool() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "use the selected tool",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_weather",
                    "parameters": {"type": "object", "properties": {}}
                },
                {
                    "type": "function",
                    "name": "lookup_news",
                    "parameters": {"type": "object", "properties": {}}
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_weather"}
            },
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let raw = openai_chat_request_json(&execution.chat_request, false);

        assert_eq!(raw["tool_choice"], "required");
        assert_eq!(raw["tools"].as_array().unwrap().len(), 1);
        assert_eq!(raw["tools"][0]["function"]["name"], "lookup_weather");
        assert!(execution.tool_kinds.contains_key("lookup_weather"));
        assert!(!execution.tool_kinds.contains_key("lookup_news"));
    }

    #[tokio::test]
    async fn responses_unknown_read_file_call_uses_available_exec_command_tool() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "inspect a file",
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"}
                    },
                    "required": ["cmd"]
                }
            }],
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let raw_completion = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/app/decomp.c\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7
            }
        });
        let response =
            openai_responses_response_from_chat_completion(&execution, raw_completion.clone());

        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["name"], "exec_command");
        assert!(
            response["output"][0]["arguments"]
                .as_str()
                .unwrap()
                .contains("cat -- '/app/decomp.c'")
        );

        let messages = response_conversation_messages_from_chat_completion(
            &execution,
            execution.conversation_messages.clone(),
            &raw_completion,
        );
        let stored_call = &messages.last().unwrap().tool_calls[0];
        assert_eq!(stored_call.function.name, "exec_command");
        assert_eq!(
            stored_call.function.arguments["cmd"],
            "cat -- '/app/decomp.c'"
        );
    }

    #[tokio::test]
    async fn responses_text_tool_call_markup_returns_function_call_item() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "read a file",
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}}
                }
            }],
            "reasoning": {"effort": "none"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();
        let response = openai_responses_response_from_chat_completion(
            &execution,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<|tool_call>call:exec_command{cmd:<|\"|>cat /app/data.txt<|\"|>}<tool_call|>"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
        );

        assert_eq!(response["output_text"], "");
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["name"], "exec_command");
        assert!(
            response["output"][0]["arguments"]
                .as_str()
                .unwrap()
                .contains("cat /app/data.txt")
        );
    }

    #[tokio::test]
    async fn responses_top_level_reasoning_effort_controls_chat_execution_policy() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "responses effort",
            "reasoning_effort": "xhigh",
            "reasoning": {"summary": "auto"}
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let effort = openai_reasoning_effort(&execution.chat_request).unwrap();
        let provider_options = openai_provider_options(&execution.chat_request);

        assert_eq!(effort, ReasoningEffort::XHigh);
        assert!(provider_options.get("reasoning_effort").is_none());
        assert_eq!(provider_options["reasoning"]["summary"], "auto");
    }

    #[tokio::test]
    async fn responses_preserve_current_sdk_provider_options() {
        let request = parse_openai_responses_request(serde_json::json!({
            "model": "mock",
            "input": "options",
            "service_tier": "priority",
            "safety_identifier": "shared-user",
            "prompt_cache_key": "cache-key",
            "prompt_cache_retention": "24h",
            "text": {
                "verbosity": "low",
                "format": {"type": "json_object"}
            }
        }))
        .unwrap();
        let execution = prepare_openai_responses_execution(
            &ResponsesStore::new(None),
            request,
            &RequestContext::default(),
        )
        .await
        .unwrap();

        let options = openai_provider_options(&execution.chat_request);
        assert_eq!(options["service_tier"], "priority");
        assert_eq!(options["safety_identifier"], "shared-user");
        assert_eq!(options["prompt_cache_key"], "cache-key");
        assert_eq!(options["prompt_cache_retention"], "24h");
        assert_eq!(options["verbosity"], "low");
        assert_eq!(options["response_format"]["type"], "json_object");
    }

    #[tokio::test]
    async fn responses_background_runs_asynchronously_and_can_be_polled() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "input": "background",
                        "background": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let created = response_json(response).await;
        assert_eq!(created["background"], true);
        assert_eq!(created["status"], "queued");
        let response_id = created["id"].as_str().unwrap();

        let mut completed = None;
        for _ in 0..100 {
            let response = tower::ServiceExt::oneshot(
                app.clone(),
                axum::http::Request::builder()
                    .uri(format!("/v1/responses/{response_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            let value = response_json(response).await;
            if value["status"] == "completed" {
                completed = Some(value);
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let completed = completed.expect("background response should complete");
        assert_eq!(completed["background"], true);
        assert!(
            completed["output_text"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        );
    }

    #[tokio::test]
    async fn responses_background_cancel_aborts_in_flight_execution() {
        let app = build_router_with_provider(Arc::new(ConcurrentProbeProvider::default()));
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "input": "cancel background",
                        "background": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        let created = response_json(response).await;
        let response_id = created["id"].as_str().unwrap();
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri(format!("/v1/responses/{response_id}/cancel"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response_json(response).await["status"], "cancelled");
        tokio::time::sleep(Duration::from_millis(150)).await;
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .uri(format!("/v1/responses/{response_id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response_json(response).await["status"], "cancelled");
    }

    #[tokio::test]
    async fn routes_openai_responses_stream_emits_response_events() {
        let app = build_router();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "stream": true,
                        "input": "hello stream responses"
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
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.output_text.done"));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn routes_openai_responses_websocket_accepts_codex_response_create() {
        use futures::SinkExt;
        use tokio_tungstenite::connect_async;
        use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let (mut websocket, response) = connect_async(format!("ws://{addr}/v1/responses"))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SWITCHING_PROTOCOLS
        );

        websocket
            .send(TungsteniteMessage::Text(
                serde_json::json!({
                    "type": "response.create",
                    "model": "mock",
                    "input": [{
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "websocket hello"}]
                    }],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": true,
                    "store": true,
                    "stream": true,
                    "include": []
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

        let mut saw_created = false;
        let mut saw_completed = false;
        while let Some(message) = websocket.next().await {
            let message = message.unwrap();
            let TungsteniteMessage::Text(text) = message else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            match value.get("type").and_then(|value| value.as_str()) {
                Some("response.created") => saw_created = true,
                Some("response.completed") => {
                    saw_completed = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_created);
        assert!(saw_completed);
        server.abort();
    }

    #[tokio::test]
    async fn routes_openai_responses_websocket_prewarm_stores_turn_state() {
        use futures::SinkExt;
        use tokio_tungstenite::connect_async;
        use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_router();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let (mut websocket, response) = connect_async(format!("ws://{addr}/v1/responses"))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SWITCHING_PROTOCOLS
        );

        websocket
            .send(TungsteniteMessage::Text(
                serde_json::json!({
                    "type": "response.create",
                    "model": "mock",
                    "input": [{
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "websocket prewarm hello"}]
                    }],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": true,
                    "store": false,
                    "stream": true,
                    "include": [],
                    "generate": false
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

        let mut previous_response_id = None;
        while let Some(message) = websocket.next().await {
            let TungsteniteMessage::Text(text) = message.unwrap() else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_ne!(
                value.get("type").and_then(|value| value.as_str()),
                Some("error")
            );
            if value.get("type").and_then(|value| value.as_str()) == Some("response.completed") {
                previous_response_id = value
                    .get("response")
                    .and_then(|response| response.get("id"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                break;
            }
        }

        let previous_response_id = previous_response_id.unwrap();
        websocket
            .send(TungsteniteMessage::Text(
                serde_json::json!({
                    "type": "response.create",
                    "model": "mock",
                    "previous_response_id": previous_response_id,
                    "input": [],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": true,
                    "store": false,
                    "stream": true,
                    "include": []
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

        let mut saw_completed = false;
        while let Some(message) = websocket.next().await {
            let TungsteniteMessage::Text(text) = message.unwrap() else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_ne!(
                value.get("type").and_then(|value| value.as_str()),
                Some("error")
            );
            if value.get("type").and_then(|value| value.as_str()) == Some("response.completed") {
                saw_completed = true;
                break;
            }
        }

        assert!(saw_completed);
        server.abort();
    }

    #[tokio::test]
    async fn routes_openai_responses_tool_call_and_previous_response_roundtrip() {
        let app = build_router();
        let first = post_json(
            app.clone(),
            "/v1/responses",
            serde_json::json!({
                "model": "mock",
                "input": "please use a tool",
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "description": "lookup a value",
                    "parameters": {"type": "object"}
                }]
            }),
        )
        .await;

        assert_eq!(first["output"][0]["type"], "function_call");
        assert_eq!(first["output"][0]["name"], "lookup");
        assert_eq!(first["output"][0]["call_id"], "call-1");

        let second = post_json(
            app,
            "/v1/responses",
            serde_json::json!({
                "model": "mock",
                "previous_response_id": first["id"].as_str().unwrap(),
                "input": [{
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "{\"answer\":\"42\"}"
                }]
            }),
        )
        .await;

        assert_eq!(second["status"], "completed");
        assert!(second["output_text"].as_str().unwrap().contains("42"));
        assert_eq!(second["previous_response_id"], first["id"]);
    }

    #[tokio::test]
    async fn routes_openai_responses_retrieve_list_input_items_and_delete() {
        let app = build_router();
        let created = post_json(
            app.clone(),
            "/v1/responses",
            serde_json::json!({
                "model": "mock",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "inspect this image"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                    ]
                }]
            }),
        )
        .await;
        let id = created["id"].as_str().unwrap();

        let retrieved_response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(retrieved_response.status(), axum::http::StatusCode::OK);
        let retrieved = response_json(retrieved_response).await;
        assert_eq!(retrieved["id"], created["id"]);

        let input_items_response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{id}/input_items"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(input_items_response.status(), axum::http::StatusCode::OK);
        let input_items = response_json(input_items_response).await;
        assert_eq!(input_items["object"], "list");
        assert_eq!(input_items["data"][0]["type"], "message");

        let list_response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("GET")
                .uri("/v1/responses")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(list_response.status(), axum::http::StatusCode::OK);
        let list = response_json(list_response).await;
        assert!(
            list["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|response| response["id"] == created["id"])
        );

        let delete_response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("DELETE")
                .uri(format!("/v1/responses/{id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(delete_response.status(), axum::http::StatusCode::OK);
        let deleted = response_json(delete_response).await;
        assert_eq!(deleted["id"], created["id"]);
        assert_eq!(deleted["object"], "response.deleted");
        assert_eq!(deleted["deleted"], true);

        let missing_response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(missing_response.status(), axum::http::StatusCode::NOT_FOUND);
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

        assert_eq!(value["usage"]["prompt_tokens"], 306);
        assert_eq!(value["usage"]["completion_tokens"], 414);
        assert_eq!(value["usage"]["total_tokens"], 720);
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

        assert_eq!(value["usage"]["input_tokens"], 306);
        assert_eq!(value["usage"]["output_tokens"], 414);
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
                    task_id: TaskId::from("model-child-01"),
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
                    task_id: TaskId::from("model-child-02"),
                    role: AgentRole::Worker,
                    text_outputs: vec!["測試完成。".to_string()],
                    tool_calls: vec![],
                    usage: ProviderUsage {
                        input_tokens: 90,
                        output_tokens: 10,
                    },
                },
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("model-child-03"),
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
                KernelTraceEvent::AgentOutput {
                    task_id: TaskId::from("model-child-04"),
                    role: AgentRole::Worker,
                    text_outputs: vec![
                        "Useful finding before a tool trace.\nTool call call_123 (exec_command) arguments: {\"cmd\":\"cat /app/secret\"}\nUseful finding after the tool trace.".to_string(),
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
            "  - model-child-01 (Worker agent): summary: 確認測試完成; confidence: high"
        ));
        assert!(reasoning.contains("  - model-child-02 (Worker agent): 測試完成。"));
        assert!(reasoning.contains("  - model-child-03 (Worker agent): intent: 確認測試完成"));
        assert!(reasoning.contains(
            "  - model-child-04 (Worker agent): Useful finding before a tool trace. Useful finding after the tool trace."
        ));
        assert!(!reasoning.contains("[no internal tool calls"));
        assert!(!reasoning.contains("tokens in/out"));
        assert!(!reasoning.contains("{\"summary\""));
        assert!(!reasoning.contains(r#"{ "intent""#));
        assert!(!reasoning.contains("exec_command"));
        assert!(!reasoning.contains("/app/secret"));
        assert!(!reasoning.contains("call_123"));
    }

    #[test]
    fn public_reasoning_summary_uses_only_latest_spawn_plan() {
        let output = KernelOutput {
            final_text: "final answer".to_string(),
            task_graph: TaskGraph::new(TaskId::from("root")),
            verification: VerificationReport {
                request_id: RequestId::from("request-a"),
                passed: false,
                issues: vec![],
                artifact_coverage: vec![],
                unresolved_tool_calls: vec![],
                budget_summary: BudgetSummary::default(),
            },
            tool_calls: vec![],
            encrypted_subagent_state: vec![],
            usage: ProviderUsage::default(),
            provider_call_count: 3,
            trace_events: vec![
                KernelTraceEvent::SpawnPlan {
                    task_id: TaskId::from("orchestration-planner"),
                    reason: "under-target repair attempt".to_string(),
                    children: public_summary_test_children("repair-child", 31),
                },
                KernelTraceEvent::SpawnPlan {
                    task_id: TaskId::from("orchestration-planner-repair"),
                    reason: "final accepted plan".to_string(),
                    children: public_summary_test_children("final-child", 32),
                },
            ],
        };

        let reasoning = public_reasoning_summary(&output);

        assert!(reasoning.contains("Orchestrator scheduled 32 bounded child agent"));
        assert!(reasoning.contains("final-child-00"));
        assert!(reasoning.contains("final accepted plan"));
        assert!(!reasoning.contains("Orchestrator scheduled 31 bounded child agent"));
        assert!(!reasoning.contains("repair-child-00"));
        assert!(!reasoning.contains("under-target repair attempt"));
    }

    fn public_summary_test_children(prefix: &str, count: usize) -> Vec<SubtaskSpec> {
        (0..count)
            .map(|index| SubtaskSpec {
                task_id: TaskId::from(format!("{prefix}-{index:02}")),
                parent_task_id: Some(TaskId::from("root")),
                spawn_depth: 1,
                role: AgentRole::Worker,
                objective: format!("worker objective {index}"),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits::default(),
            })
            .collect()
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
                std::sync::Arc::new(MockProvider),
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
                "messages": [{"role": "user", "content": "spawn record this"}]
            }),
        )
        .await;

        let content = std::fs::read_to_string(&path).unwrap();
        let line = content.lines().next().unwrap();
        let example: serde_json::Value = serde_json::from_str(line).unwrap();
        let conversations = example["conversations"].as_array().unwrap();
        assert_eq!(conversations[0]["from"], "human");
        assert_eq!(conversations[0]["value"], "spawn record this");
        assert!(conversations.iter().any(|turn| {
            turn["from"] == "function_call"
                && turn["value"].as_str().unwrap().contains("spawn_agent")
        }));
        assert_eq!(conversations.last().unwrap()["from"], "gpt");
        assert_eq!(
            conversations.last().unwrap()["value"],
            "Here is a clear, usable answer based on the verified agent results."
        );

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
    async fn orchestration_sse_emits_heartbeats_while_a_slow_model_is_running() {
        let metrics = RuntimeMetrics::default();
        let response = orchestration_sse_response_with_interval(
            async {
                tokio::time::sleep(Duration::from_millis(30)).await;
                sse_response("data: {\"done\":true}\n\n".to_string())
            },
            metrics.clone(),
            Duration::from_millis(5),
        );

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let body = response_text(response).await;
        assert!(body.starts_with(": miya orchestration active\n\n"));
        assert!(body.matches(": miya orchestration active").count() >= 2);
        assert!(body.contains("data: {\"done\":true}"));
        assert!(
            metrics
                .orchestration_stream_heartbeats
                .load(Ordering::Relaxed)
                >= 1
        );
        assert_eq!(
            metrics.orchestration_streams_active.load(Ordering::Relaxed),
            0
        );
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
        assert!(body.contains("Orchestrator scheduled 16 bounded child agent"));
        assert!(body.contains("child-00 (Worker agent): summarized worker finding"));
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
        assert!(reasoning.contains("Orchestrator scheduled 16 bounded child agent"));
        assert!(reasoning.contains("Agent output summaries:"));
        assert!(reasoning.contains("child-00 (Worker agent): summarized worker finding"));
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
        assert!(body.contains("child-00 (Worker agent): summarized worker finding"));
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
        assert_eq!(provider.invoke_calls(), 19);
        assert_eq!(provider.stream_calls(), 0);
        assert!(body.contains("\"reasoning_content\":\"Multi-agent process summary"));
        assert!(body.contains("Orchestrator scheduled 16 bounded child agent"));
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
        assert_eq!(provider.invoke_calls(), 19);
        assert_eq!(provider.stream_calls(), 0);
        assert!(body.contains("\"type\":\"thinking_delta\""));
        assert!(body.contains("Multi-agent process summary"));
        assert!(body.contains("Orchestrator scheduled 16 bounded child agent"));
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
    async fn routes_do_not_expose_nonstandard_batch_aliases() {
        for path in [
            "/chat/completions/batch",
            "/v1/chat/completions/batch",
            "/v1/v1/chat/completions/batch",
            "/messages/batch",
            "/v1/messages/batch",
            "/v1/v1/messages/batch",
        ] {
            let response = tower::ServiceExt::oneshot(
                build_router(),
                axum::http::Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{\"requests\":[]}"))
                    .unwrap(),
            )
            .await
            .unwrap();

            assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
        }
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
    async fn routes_openai_files_and_batch_execute_durable_jsonl_workflow() {
        let app = build_router();
        let boundary = "miya-test-boundary";
        let jsonl = serde_json::json!({
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "mock",
                "reasoning_effort": "none",
                "messages": [{"role": "user", "content": "official batch input"}]
            }
        })
        .to_string();
        let multipart = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"batch.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n{jsonl}\n\r\n--{boundary}--\r\n"
        );
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(axum::body::Body::from(multipart))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let file = response_json(response).await;
        assert_eq!(file["purpose"], "batch");
        let file_id = file["id"].as_str().unwrap();

        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "input_file_id": file_id,
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let created = response_json(response).await;
        let batch_id = created["id"].as_str().unwrap();
        let mut completed = None;
        for _ in 0..100 {
            let response = tower::ServiceExt::oneshot(
                app.clone(),
                axum::http::Request::builder()
                    .uri(format!("/v1/batches/{batch_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            let batch = response_json(response).await;
            if batch["status"] == "completed" {
                completed = Some(batch);
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let completed = completed.expect("OpenAI batch should complete");
        assert_eq!(completed["request_counts"]["completed"], 1);
        let output_file_id = completed["output_file_id"].as_str().unwrap();
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .uri(format!("/v1/files/{output_file_id}/content"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let output = String::from_utf8(body.to_vec()).unwrap();
        assert!(output.contains("\"custom_id\":\"request-1\""));
        assert!(output.contains("official batch input"));
    }

    #[tokio::test]
    async fn routes_official_anthropic_batch_returns_jsonl_results() {
        let app = build_router_with_provider(std::sync::Arc::new(InputEchoProvider));
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batches")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"custom_id": "alpha", "params": {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "alpha anthropic"}]}},
                            {"custom_id": "beta", "params": {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "beta anthropic"}]}}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let created = response_json(response).await;
        assert_eq!(created["type"], "message_batch");
        let batch_id = created["id"].as_str().unwrap();
        let mut ended = false;
        for _ in 0..100 {
            let response = tower::ServiceExt::oneshot(
                app.clone(),
                axum::http::Request::builder()
                    .uri(format!("/v1/messages/batches/{batch_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            let value = response_json(response).await;
            if value["processing_status"] == "ended" {
                ended = true;
                assert_eq!(value["request_counts"]["succeeded"], 2);
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(ended);
        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .uri(format!("/v1/messages/batches/{batch_id}/results"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let lines = String::from_utf8(body.to_vec()).unwrap();
        assert!(lines.contains("\"custom_id\":\"alpha\""));
        assert!(lines.contains("alpha anthropic"));
        assert!(lines.contains("\"custom_id\":\"beta\""));
        assert!(lines.contains("beta anthropic"));
    }

    #[tokio::test]
    async fn routes_official_anthropic_batch_runs_requests_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider(provider.clone());
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batches")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"custom_id": "first", "params": {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "first anthropic"}]}},
                            {"custom_id": "second", "params": {"model": "mock", "max_tokens": 256, "messages": [{"role": "user", "content": "second anthropic"}]}}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let created = response_json(response).await;
        let batch_id = created["id"].as_str().unwrap();
        for _ in 0..100 {
            let response = tower::ServiceExt::oneshot(
                app.clone(),
                axum::http::Request::builder()
                    .uri(format!("/v1/messages/batches/{batch_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            if response_json(response).await["processing_status"] == "ended" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(provider.max_in_flight() >= 2);
    }

    #[tokio::test]
    async fn resilient_provider_retries_429_and_records_prometheus_metrics() {
        let inner = Arc::new(FlakyRateLimitProvider::default());
        let metrics = RuntimeMetrics::default();
        let provider = ResilientProvider {
            inner,
            config: ResilienceConfig {
                max_retries: 2,
                base_delay: Duration::from_millis(1),
                circuit_failure_threshold: 5,
                circuit_cooldown: Duration::from_millis(10),
            },
            circuit: CircuitBreaker::default(),
            metrics: metrics.clone(),
        };
        let request = provider_probe_request();
        provider.invoke(request).await.unwrap();
        assert_eq!(metrics.provider_attempts.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.provider_retries.load(Ordering::Relaxed), 2);
        assert!(
            metrics
                .prometheus()
                .contains("miya_provider_retries_total 2")
        );
    }

    #[tokio::test]
    async fn provider_admission_limits_concurrency_across_requests() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let managed: Arc<dyn ModelProvider> = Arc::new(AdmissionProvider {
            inner: provider.clone(),
            admission: ProviderAdmission {
                semaphore: Some(Arc::new(Semaphore::new(1))),
                wait_timeout: Duration::from_secs(5),
            },
        });
        let app = build_router_with_provider(managed);
        let request = |text: &str| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "messages": [{"role": "user", "content": text}]
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        let (first, second) = tokio::join!(
            tower::ServiceExt::oneshot(app.clone(), request("first")),
            tower::ServiceExt::oneshot(app, request("second")),
        );

        assert_eq!(first.unwrap().status(), axum::http::StatusCode::OK);
        assert_eq!(second.unwrap().status(), axum::http::StatusCode::OK);
        assert_eq!(provider.max_in_flight(), 1);
    }

    #[tokio::test]
    async fn routes_anthropic_batch_limits_same_tenant_concurrency() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let response = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batches")
                .header("content-type", "application/json")
                .header("x-tenant-id", "tenant-a")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [
                            {"custom_id": "first", "params": {"model": "mock", "max_tokens": 256, "metadata": {"max_parallel_agents": 1}, "messages": [{"role": "user", "content": "first anthropic"}]}},
                            {"custom_id": "second", "params": {"model": "mock", "max_tokens": 256, "metadata": {"max_parallel_agents": 1}, "messages": [{"role": "user", "content": "second anthropic"}]}}
                        ]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let created = response_json(response).await;
        let batch_id = created["id"].as_str().unwrap();
        for _ in 0..100 {
            let response = tower::ServiceExt::oneshot(
                app.clone(),
                axum::http::Request::builder()
                    .uri(format!("/v1/messages/batches/{batch_id}"))
                    .header("x-tenant-id", "tenant-a")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            if response_json(response).await["processing_status"] == "ended" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(provider.max_in_flight(), 1);
    }

    #[tokio::test]
    async fn routes_anthropic_batch_allows_different_tenants_to_run_concurrently() {
        let provider = std::sync::Arc::new(ConcurrentProbeProvider::default());
        let app = build_router_with_provider_and_tenant_limit(provider.clone(), 1);
        let request = |tenant: &str, custom_id: &str| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/messages/batches")
                .header("content-type", "application/json")
                .header("x-tenant-id", tenant)
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "requests": [{
                            "custom_id": custom_id,
                            "params": {
                                "model": "mock",
                                "max_tokens": 256,
                                "messages": [{"role": "user", "content": custom_id}]
                            }
                        }]
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        let (first, second) = tokio::join!(
            tower::ServiceExt::oneshot(app.clone(), request("tenant-a", "first")),
            tower::ServiceExt::oneshot(app, request("tenant-b", "second")),
        );
        assert_eq!(first.unwrap().status(), axum::http::StatusCode::OK);
        assert_eq!(second.unwrap().status(), axum::http::StatusCode::OK);
        for _ in 0..100 {
            if provider.max_in_flight() >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
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

    #[derive(Debug)]
    struct InputEchoProvider;

    #[async_trait::async_trait]
    impl provider_core::ModelProvider for InputEchoProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, provider_core::ProviderError> {
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                return Ok(route_spawn_plan_response(
                    request,
                    "echo provider model-selected context workers",
                ));
            }
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
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                let mut response = route_spawn_plan_response(
                    request,
                    "usage provider model-selected accounting workers",
                );
                response.usage = ProviderUsage {
                    input_tokens: 17,
                    output_tokens: 23,
                };
                return Ok(response);
            }
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
            *self.provider_options.lock().unwrap() = Some(request.provider_options.clone());
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                return Ok(route_spawn_plan_response(
                    request,
                    "provider-options probe model-selected workers",
                ));
            }
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

    #[derive(Default)]
    struct FlakyRateLimitProvider {
        attempts: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ModelProvider for FlakyRateLimitProvider {
        async fn invoke(
            &self,
            request: provider_core::ProviderRequest,
        ) -> Result<provider_core::ProviderResponse, ProviderError> {
            let attempt = self
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if attempt < 2 {
                return Err(ProviderError::Http {
                    provider: "test".to_string(),
                    status: 429,
                    code: Some("rate_limit_exceeded".to_string()),
                    message: "slow down".to_string(),
                    retry_after_ms: Some(1),
                });
            }
            Ok(provider_core::ProviderResponse {
                artifacts: vec![AgentArtifact::Text {
                    id: ArtifactId::from("flaky-success"),
                    scope: request.scope,
                    text: "ok".to_string(),
                }],
                tool_calls: Vec::new(),
                usage: ProviderUsage::default(),
            })
        }
    }

    fn provider_probe_request() -> provider_core::ProviderRequest {
        provider_core::ProviderRequest {
            scope: IsolationKey::new("tenant", "request", "conversation"),
            task: SubtaskSpec {
                task_id: TaskId::from("root"),
                parent_task_id: None,
                spawn_depth: 0,
                role: AgentRole::Leader,
                objective: "probe".to_string(),
                input_artifact_refs: Vec::new(),
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text]),
                limits: AgentLimits::default(),
            },
            model: "mock".to_string(),
            system_instructions: Vec::new(),
            thinking_enabled: false,
            thinking_format: ThinkingFormat::Auto,
            input_text: "probe".to_string(),
            messages: Vec::new(),
            media_artifacts: Vec::new(),
            artifacts: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            tool_results: Vec::new(),
            provider_options: serde_json::json!({}),
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

            if request.task.objective.contains("orchestration plan") {
                return Ok(route_spawn_plan_response(
                    request,
                    "concurrency probe model-selected workers",
                ));
            }
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
                    let target = route_target_parallel_agents(&request.system_instructions);
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

    fn route_spawn_plan_response(
        request: provider_core::ProviderRequest,
        reason: &str,
    ) -> provider_core::ProviderResponse {
        let target = route_target_parallel_agents(&request.system_instructions);
        provider_core::ProviderResponse {
            artifacts: vec![AgentArtifact::SpawnPlan {
                id: ArtifactId::from(format!("spawn-plan-{}", request.task.task_id.as_ref())),
                scope: request.scope.clone(),
                plan: SpawnPlan {
                    parent_task_id: request.task.task_id,
                    reason: reason.to_string(),
                    children: (0..target)
                        .map(|index| SubtaskSpec {
                            task_id: TaskId::from(format!("route-child-{index:02}")),
                            parent_task_id: Some(TaskId::from("root")),
                            spawn_depth: 1,
                            role: AgentRole::Worker,
                            objective: format!("route worker slice {index}"),
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
        }
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
            if request.task.role == AgentRole::Leader
                && request.task.objective.contains("orchestration plan")
            {
                return Ok(route_spawn_plan_response(
                    request,
                    "streaming probe model-selected fallback workers",
                ));
            }
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
