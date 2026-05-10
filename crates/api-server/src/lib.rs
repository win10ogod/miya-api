use std::{
    collections::{BTreeSet, VecDeque},
    convert::Infallible,
    sync::Arc,
};

use agent_kernel::{KernelOutput, KernelPolicy, KernelRunner, MockProvider};
use agent_protocol::*;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use context_store::{
    ContextAppendRecord, ContextAssembly, ContextAssemblyOptions, DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_CONTEXT_BYTES, DEFAULT_RECENT_TAIL_CHUNKS, SurrealKvContextStore,
};
use futures::{Stream, StreamExt, future::join_all};
use provider_anthropic::AnthropicProvider;
use provider_core::{ModelProvider, ProviderFinishReason, ProviderStream, ProviderStreamEvent};
use provider_openai::OpenAiProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

const MAX_BATCH_REQUESTS: usize = 64;

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
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAiChatBatchRequest {
    pub requests: Vec<serde_json::Value>,
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
        Self {
            kernel: Arc::new(KernelRunner::new(provider, KernelPolicy::default())),
            context,
            direct,
        }
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
                    .map_err(|error| error.to_string())?
                    .error_for_status()
                    .map_err(|error| error.to_string())?;
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
                    .map_err(|error| error.to_string())?
                    .error_for_status()
                    .map_err(|error| error.to_string())?;
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
    builder
        .json(&body)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|error| error.to_string())
}

fn sanitize_direct_openai_request(mut request: serde_json::Value) -> serde_json::Value {
    strip_gateway_metadata(&mut request);
    strip_gateway_reasoning_effort(&mut request);
    disable_qwen_thinking_by_default(&mut request);
    request
}

fn sanitize_direct_anthropic_request(mut request: serde_json::Value) -> serde_json::Value {
    strip_gateway_metadata(&mut request);
    strip_gateway_reasoning_effort(&mut request);
    request
}

fn strip_gateway_metadata(request: &mut serde_json::Value) {
    if let Some(object) = request.as_object_mut() {
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
        let assembly = store
            .assemble(
                normalized.tenant_id.as_ref(),
                &config.context_id,
                ContextAssemblyOptions {
                    query,
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

#[derive(Clone, Debug, Default, Serialize)]
struct ContextUsageReport {
    enabled: bool,
    context_id: Option<String>,
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
    Ok(build_router_with_state(
        AppState::with_provider_context_and_direct(provider, context, direct),
    ))
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
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/batch", post(chat_completions_batch))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/batch", post(messages_batch))
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
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "mock",
            "object": "model",
            "owned_by": "multi-agent-api"
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let mut request = match parse_openai_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = request.stream;
    request.stream = false;
    if matches!(openai_reasoning_effort(&request), Ok(ReasoningEffort::None)) {
        if stream {
            return match state.direct.openai_chat_stream(raw_request).await {
                Ok(response) => response,
                Err(error) => internal_error_response(error),
            };
        }
        return match state.direct.openai_chat(raw_request).await {
            Ok(value) => Json(value).into_response(),
            Err(error) => internal_error_response(error),
        };
    }

    let model = request.model.clone();
    let tool_response_format = openai_tool_response_format(&request);
    let mut normalized = match normalize_openai_chat(request) {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);

    if stream && !requires_full_orchestration_before_stream(&normalized) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => {
                format_openai_provider_stream_response(model, provider_stream, tool_response_format)
            }
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
            if stream {
                return format_openai_stream_response(model, output, tool_response_format);
            }
            let mut response = format_openai_response(
                model,
                output,
                include_encrypted_state,
                tool_response_format,
            );
            attach_context_report(&mut response, prepared_context.as_ref(), context_report);
            Json(response).into_response()
        }
        Err(error) => internal_error_response(error.to_string()),
    }
}

async fn chat_completions_batch(
    State(state): State<AppState>,
    Json(batch): Json<OpenAiChatBatchRequest>,
) -> Response {
    if let Err(error) = validate_batch_len(batch.requests.len()) {
        return api_error_response(error);
    }

    let kernel = state.kernel.clone();
    let context = state.context.clone();
    let direct = state.direct.clone();
    let data = join_all(
        batch
            .requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let kernel = kernel.clone();
                let context = context.clone();
                let direct = direct.clone();
                async move { run_openai_batch_item(kernel, context, direct, index, request).await }
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
    Json(raw_request): Json<serde_json::Value>,
) -> Response {
    let mut request = match parse_anthropic_request(raw_request.clone()) {
        Ok(request) => request,
        Err(error) => return api_error_response(error),
    };
    let stream = request.stream;
    request.stream = false;
    if matches!(
        anthropic_reasoning_effort(&request),
        Ok(ReasoningEffort::None)
    ) {
        if stream {
            return match state.direct.anthropic_messages_stream(raw_request).await {
                Ok(response) => response,
                Err(error) => internal_error_response(error),
            };
        }
        return match state.direct.anthropic_messages(raw_request).await {
            Ok(value) => Json(value).into_response(),
            Err(error) => internal_error_response(error),
        };
    }

    let model = request.model.clone();
    let mut normalized = match normalize_anthropic_messages(request) {
        Ok(normalized) => normalized,
        Err(error) => return api_error_response(error),
    };
    let prepared_context = match state.context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return internal_error_response(error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);

    if stream && !requires_full_orchestration_before_stream(&normalized) {
        return match state.kernel.stream_root(normalized).await {
            Ok(provider_stream) => {
                format_anthropic_provider_stream_response(model, provider_stream)
            }
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
            if stream {
                return format_anthropic_stream_response(model, output);
            }
            let mut response = format_anthropic_response(model, output, include_encrypted_state);
            attach_context_report(&mut response, prepared_context.as_ref(), context_report);
            Json(response).into_response()
        }
        Err(error) => internal_error_response(error.to_string()),
    }
}

async fn messages_batch(
    State(state): State<AppState>,
    Json(batch): Json<AnthropicMessagesBatchRequest>,
) -> Response {
    if let Err(error) = validate_batch_len(batch.requests.len()) {
        return api_error_response(error);
    }

    let kernel = state.kernel.clone();
    let context = state.context.clone();
    let direct = state.direct.clone();
    let data =
        join_all(
            batch
                .requests
                .into_iter()
                .enumerate()
                .map(|(index, request)| {
                    let kernel = kernel.clone();
                    let context = context.clone();
                    let direct = direct.clone();
                    async move {
                        run_anthropic_batch_item(kernel, context, direct, index, request).await
                    }
                }),
        )
        .await;

    Json(serde_json::json!({
        "type": "message_batch",
        "data": data
    }))
    .into_response()
}

async fn run_openai_batch_item(
    kernel: Arc<KernelRunner<Arc<dyn ModelProvider>>>,
    context: ApiContextManager,
    direct: DirectBackend,
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
    if matches!(openai_reasoning_effort(&request), Ok(ReasoningEffort::None)) {
        return match direct.openai_chat(raw_request).await {
            Ok(response) => batch_success(index, response),
            Err(error) => batch_kernel_error(index, error),
        };
    }

    let model = request.model.clone();
    let tool_response_format = openai_tool_response_format(&request);
    let mut normalized = match normalize_openai_chat(request) {
        Ok(normalized) => normalized,
        Err(error) => return batch_api_error(index, error),
    };
    let prepared_context = match context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return batch_kernel_error(index, error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);

    match kernel.run(normalized).await {
        Ok(output) => match finalize_context_report(&context, &prepared_context, &output).await {
            Ok(context_report) => {
                let mut response = format_openai_response(
                    model,
                    output,
                    include_encrypted_state,
                    tool_response_format,
                );
                attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                batch_success(index, response)
            }
            Err(error) => batch_kernel_error(index, error),
        },
        Err(error) => batch_kernel_error(index, error.to_string()),
    }
}

async fn run_anthropic_batch_item(
    kernel: Arc<KernelRunner<Arc<dyn ModelProvider>>>,
    context: ApiContextManager,
    direct: DirectBackend,
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
    if matches!(
        anthropic_reasoning_effort(&request),
        Ok(ReasoningEffort::None)
    ) {
        return match direct.anthropic_messages(raw_request).await {
            Ok(response) => batch_success(index, response),
            Err(error) => batch_kernel_error(index, error),
        };
    }

    let model = request.model.clone();
    let mut normalized = match normalize_anthropic_messages(request) {
        Ok(normalized) => normalized,
        Err(error) => return batch_api_error(index, error),
    };
    let prepared_context = match context.prepare(&mut normalized).await {
        Ok(prepared) => prepared,
        Err(error) => return batch_kernel_error(index, error),
    };
    let include_encrypted_state = include_encrypted_subagent_state(&normalized.metadata);

    match kernel.run(normalized).await {
        Ok(output) => match finalize_context_report(&context, &prepared_context, &output).await {
            Ok(context_report) => {
                let mut response =
                    format_anthropic_response(model, output, include_encrypted_state);
                attach_context_report(&mut response, prepared_context.as_ref(), context_report);
                batch_success(index, response)
            }
            Err(error) => batch_kernel_error(index, error),
        },
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

fn format_openai_response(
    model: String,
    output: KernelOutput,
    include_encrypted_state: bool,
    tool_response_format: OpenAiToolResponseFormat,
) -> serde_json::Value {
    let encrypted_state = output.encrypted_subagent_state.clone();
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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
    };

    if include_encrypted_state {
        value["encrypted_agent_state"] = serde_json::json!(encrypted_state);
    }

    value
}

fn format_openai_stream_response(
    model: String,
    output: KernelOutput,
    tool_response_format: OpenAiToolResponseFormat,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let mut body = String::new();
    body.push_str(&openai_stream_role_chunk(&id, &model));

    if output.verification.passed {
        body.push_str(&sse_data(openai_stream_chunk(
            &id,
            &model,
            serde_json::json!({"content": output.final_text}),
            serde_json::Value::Null,
        )));
        body.push_str(&openai_stream_finish_chunk(&id, &model, "stop"));
    } else if tool_response_format == OpenAiToolResponseFormat::LegacyFunctions {
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

fn format_openai_provider_stream_response(
    model: String,
    provider_stream: ProviderStream,
    tool_response_format: OpenAiToolResponseFormat,
) -> Response {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let state = OpenAiStreamState {
        id,
        model,
        provider_stream,
        tool_response_format,
        pending: VecDeque::from([openai_stream_role_chunk_placeholder()]),
        finished: false,
        saw_tool_delta: false,
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
                }
                None => {
                    state.pending.push_back(openai_stream_finish_chunk(
                        &state.id,
                        &state.model,
                        "stop",
                    ));
                    state.pending.push_back("data: [DONE]\n\n".to_string());
                    state.finished = true;
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
    pending: VecDeque<String>,
    finished: bool,
    saw_tool_delta: bool,
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
            }
            ProviderStreamEvent::Usage { .. } | ProviderStreamEvent::TextDelta { .. } => {}
        }
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

fn requires_full_orchestration_before_stream(request: &NormalizedRequest) -> bool {
    if include_encrypted_subagent_state(&request.metadata)
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
) -> serde_json::Value {
    let encrypted_state = output.encrypted_subagent_state.clone();
    let mut value = if output.verification.passed {
        serde_json::json!({
            "id": format!("msg_{}", Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": output.final_text
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        })
    } else {
        serde_json::json!({
            "id": format!("msg_{}", Uuid::new_v4()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": output.tool_calls.into_iter().map(anthropic_tool_use_json).collect::<Vec<_>>(),
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        })
    };

    if include_encrypted_state {
        value["encrypted_agent_state"] = serde_json::json!(encrypted_state);
    }

    value
}

fn format_anthropic_stream_response(model: String, output: KernelOutput) -> Response {
    let id = format!("msg_{}", Uuid::new_v4());
    let mut body = String::new();
    body.push_str(&anthropic_message_start_event(&id, &model));

    if output.verification.passed {
        body.push_str(&sse_event(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        ));
        body.push_str(&sse_event(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": output.final_text}
            }),
        ));
        body.push_str(&anthropic_content_block_stop_event(0));
        body.push_str(&anthropic_message_delta_event("end_turn"));
    } else {
        for (index, call) in output.tool_calls.into_iter().enumerate() {
            body.push_str(&anthropic_tool_use_events(index, call));
        }
        body.push_str(&anthropic_message_delta_event("tool_use"));
    }

    body.push_str(&sse_event(
        "message_stop",
        serde_json::json!({"type": "message_stop"}),
    ));
    sse_response(body)
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
) -> Response {
    let id = format!("msg_{}", Uuid::new_v4());
    let state = AnthropicStreamState {
        id,
        model,
        provider_stream,
        pending: VecDeque::from([anthropic_message_start_placeholder()]),
        text_block_open: false,
        tool_blocks_open: BTreeSet::new(),
        finished: false,
        saw_tool_delta: false,
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
    pending: VecDeque<String>,
    text_block_open: bool,
    tool_blocks_open: BTreeSet<usize>,
    finished: bool,
    saw_tool_delta: bool,
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
            ProviderStreamEvent::Usage { .. } | ProviderStreamEvent::TextDelta { .. } => {}
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
    if request.stream {
        return Err(ApiError::StreamUnsupported);
    }

    let request_id = RequestId::from(Uuid::new_v4().to_string());
    let tenant_id = TenantId::from("default");
    let conversation_fingerprint = ConversationFingerprint::from(sha256_hex(
        &serde_json::to_string(&request.messages_debug()).unwrap_or_default(),
    ));
    let scope = IsolationKey {
        tenant_id: tenant_id.clone(),
        request_id: request_id.clone(),
        conversation_fingerprint: conversation_fingerprint.clone(),
    };

    let mut media_artifacts = Vec::new();
    let mut normalized_messages = Vec::new();
    let mut tool_results = Vec::new();
    let thinking_enabled = openai_thinking_enabled(&request);
    let thinking_format = thinking_format(&request.model, &request.metadata);
    let reasoning_effort = openai_reasoning_effort(&request)?;
    let tool_choice = openai_effective_tool_choice(&request)?;
    let parallel_tool_calls = request.parallel_tool_calls;
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
        model: request.model,
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
        metadata: request.metadata,
    })
}

pub fn normalize_anthropic_messages(
    request: AnthropicMessagesRequest,
) -> Result<NormalizedRequest, ApiError> {
    if request.stream {
        return Err(ApiError::StreamUnsupported);
    }

    let request_id = RequestId::from(Uuid::new_v4().to_string());
    let tenant_id = TenantId::from("default");
    let conversation_fingerprint = ConversationFingerprint::from(sha256_hex(
        &serde_json::to_string(&request.messages_debug()).unwrap_or_default(),
    ));
    let scope = IsolationKey {
        tenant_id: tenant_id.clone(),
        request_id: request_id.clone(),
        conversation_fingerprint: conversation_fingerprint.clone(),
    };

    let mut media_artifacts = Vec::new();
    let mut normalized_messages = Vec::new();
    let mut tool_results = Vec::new();
    let thinking_enabled = anthropic_thinking_enabled(&request);
    let thinking_format = thinking_format(&request.model, &request.metadata);
    let reasoning_effort = anthropic_reasoning_effort(&request)?;
    let (tool_choice, parallel_tool_calls) = anthropic_tool_choice(&request.tool_choice)?;

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
        model: request.model,
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
        metadata: request.metadata,
    })
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
}

fn openai_reasoning_effort(request: &OpenAiChatRequest) -> Result<ReasoningEffort, ApiError> {
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

fn anthropic_thinking_enabled(request: &AnthropicMessagesRequest) -> bool {
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
}

fn thinking_format(model: &str, metadata: &serde_json::Value) -> ThinkingFormat {
    match metadata
        .get("thinking_format")
        .and_then(|value| value.as_str())
    {
        Some("qwen_dashscope") => ThinkingFormat::QwenDashScope,
        Some("qwen_chat_template") => ThinkingFormat::QwenChatTemplate,
        Some("gemma_system_token") => ThinkingFormat::GemmaSystemToken,
        _ if model.to_lowercase().contains("gemma") => ThinkingFormat::GemmaSystemToken,
        _ if model.to_lowercase().contains("qwen") => ThinkingFormat::QwenChatTemplate,
        _ => ThinkingFormat::Auto,
    }
}

impl OpenAiChatRequest {
    fn messages_debug(&self) -> String {
        format!("{:?}", self.messages)
    }
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
        assert_eq!(normalized.reasoning_effort.max_agents(), 16);
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
            "model": "Qwen3.6-27B-FP8",
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
        assert_eq!(encrypted.len(), 1);
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
        let app = build_router();
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
                "metadata": {"include_encrypted_subagent_state": true},
                "messages": [{"role": "user", "content": "deep compare across eight independent dimensions"}]
            }),
        )
        .await;

        let low_score = coverage_score_from_openai_response(&low);
        let high_score = coverage_score_from_openai_response(&high);
        assert_eq!(low_score, 3);
        assert_eq!(high_score, 8);
        assert!(high_score > low_score);
        assert_eq!(low["encrypted_agent_state"].as_array().unwrap().len(), 3);
        assert_eq!(high["encrypted_agent_state"].as_array().unwrap().len(), 8);
    }

    #[tokio::test]
    async fn routes_openai_context_retrieves_surreal_kv_memory() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(MockProvider), context);

        let first = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "mock",
                "metadata": {
                    "context": {"id": "rp-memory", "include_report": true}
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
                        "id": "rp-memory",
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
            build_router_with_provider_and_context(std::sync::Arc::new(MockProvider), context);

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
    async fn routes_anthropic_batch_returns_isolated_responses() {
        let app = build_router();
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
    async fn routes_anthropic_context_retrieves_surreal_kv_memory() {
        let temp = tempfile::tempdir().unwrap();
        let context =
            ApiContextManager::surreal_kv(SurrealKvContextStore::open(temp.path()).unwrap());
        let app =
            build_router_with_provider_and_context(std::sync::Arc::new(MockProvider), context);

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
