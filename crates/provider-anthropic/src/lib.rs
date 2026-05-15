use std::{collections::VecDeque, pin::Pin, time::Duration};

use agent_protocol::*;
use async_trait::async_trait;
use futures::StreamExt;
use provider_core::{
    ModelProvider, ProviderError, ProviderFinishReason, ProviderRequest, ProviderResponse,
    ProviderStream, ProviderStreamEvent,
};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    api_version: String,
}

impl AnthropicProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        api_version: impl Into<String>,
    ) -> Self {
        Self {
            client: provider_http_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            api_version: api_version.into(),
        }
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ProviderError::Rejected("ANTHROPIC_API_KEY is required".to_string()))?;
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        let api_version =
            std::env::var("ANTHROPIC_VERSION").unwrap_or_else(|_| "2023-06-01".to_string());
        Ok(Self::new(base_url, api_key, api_version))
    }

    pub fn build_request_body(request: &ProviderRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "max_tokens": request.task.limits.max_tokens,
            "system": format_system_prompt(
                &request.task.role,
                &request.task.objective,
                &request.system_instructions
            ),
            "messages": anthropic_messages(request)
        });

        if request.thinking_enabled {
            body["thinking"] = anthropic_thinking_config();
        }

        if !request.tools.is_empty() {
            body["tools"] = serde_json::Value::Array(
                request
                    .tools
                    .iter()
                    .map(anthropic_tool_definition)
                    .collect(),
            );
            if let Some(tool_choice) = anthropic_tool_choice(
                &request.tool_choice,
                request.parallel_tool_calls,
                request.thinking_enabled,
            ) {
                body["tool_choice"] = tool_choice;
            }
        }

        merge_provider_options(&mut body, &request.provider_options);
        body
    }

    pub fn build_stream_request_body(request: &ProviderRequest) -> serde_json::Value {
        let mut body = Self::build_request_body(request);
        body["stream"] = serde_json::Value::Bool(true);
        body
    }

    pub fn parse_response(
        scope: &IsolationKey,
        task_id: &TaskId,
        value: serde_json::Value,
    ) -> Result<ProviderResponse, ProviderError> {
        let content = value
            .get("content")
            .and_then(|content| content.as_array())
            .ok_or_else(|| {
                ProviderError::Rejected("Anthropic response missing content".to_string())
            })?;

        let mut response = ProviderResponse::default();

        for block in content {
            match block.get("type").and_then(|kind| kind.as_str()) {
                Some("text") => {
                    if let Some(artifact) = block
                        .get("text")
                        .and_then(|text| text.as_str())
                        .and_then(|text| parse_structured_or_text(scope, task_id, text))
                    {
                        response.artifacts.push(artifact);
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|id| id.as_str())
                        .unwrap_or("tool-call");
                    let name = block
                        .get("name")
                        .and_then(|name| name.as_str())
                        .unwrap_or("unknown");
                    let arguments = block
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    response.tool_calls.push(ToolCallRecord {
                        tool_call_id: ToolCallId::from(id),
                        scope: scope.clone(),
                        task_id: task_id.clone(),
                        agent_id: AgentId::from(format!("agent-{}", task_id.as_ref())),
                        tool_name: name.to_string(),
                        arguments_sha256: sha256_hex(&arguments.to_string()),
                        arguments_json: arguments,
                        status: ToolCallStatus::Pending,
                        created_at_ms: 0,
                        resolved_at_ms: None,
                    });
                }
                _ => {}
            }
        }

        response.usage = value
            .get("usage")
            .and_then(|usage| {
                serde_json::from_value::<provider_core::ProviderUsage>(usage.clone()).ok()
            })
            .unwrap_or_default();

        Ok(response)
    }
}

fn provider_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(env_u64(
            "MIYA_PROVIDER_TIMEOUT_SECS",
            300,
        )))
        .connect_timeout(Duration::from_secs(env_u64(
            "MIYA_PROVIDER_CONNECT_TIMEOUT_SECS",
            30,
        )))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn anthropic_thinking_config() -> serde_json::Value {
    serde_json::json!({
        "type": "enabled",
        "budget_tokens": 1024
    })
}

fn merge_provider_options(body: &mut serde_json::Value, options: &serde_json::Value) {
    let (Some(body), Some(options)) = (body.as_object_mut(), options.as_object()) else {
        return;
    };
    for (key, value) in options {
        if !value.is_null() && !is_anthropic_core_field(key) {
            body.insert(key.clone(), value.clone());
        }
    }
}

fn is_anthropic_core_field(key: &str) -> bool {
    matches!(
        key,
        "model" | "system" | "messages" | "tools" | "tool_choice" | "stream"
    )
}

fn format_system_prompt(
    role: &AgentRole,
    objective: &str,
    user_system_instructions: &[String],
) -> String {
    let user_instructions = if user_system_instructions.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nUser/system instructions to preserve:\n{}",
            user_system_instructions.join("\n")
        )
    };
    let worker_contract = if matches!(role, AgentRole::Worker) {
        " Worker agents have no client-visible tools in this turn; do not request, invent, or simulate tool calls. Return a concise intermediate artifact only."
    } else {
        ""
    };
    format!(
        "You are a bounded multi-agent worker. Role: {role:?}. Objective: {objective}. If proposing child agents, return JSON with type=spawn_plan and a plan object. Final answers must be natural, directly useful, and must not expose sub-agent state, internal tool calls, or orchestration details.{worker_contract} Preserve formatting exactly when the answer contains XML/HTML-like tags, Markdown, fenced code, lists, tables, or delimiter-separated blocks. Do not minify, collapse line breaks, remove spaces around headings, or merge tag-delimited sections.{user_instructions}"
    )
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let scope = request.scope.clone();
        let task_id = request.task.task_id.clone();
        let body = Self::build_request_body(&request);
        let value = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?
            .error_for_status()
            .map_err(|error| ProviderError::Rejected(error.to_string()))?
            .json::<serde_json::Value>()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?;

        Self::parse_response(&scope, &task_id, value)
    }

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream, ProviderError> {
        let body = Self::build_stream_request_body(&request);
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?
            .error_for_status()
            .map_err(|error| ProviderError::Rejected(error.to_string()))?;

        Ok(anthropic_sse_provider_stream(response))
    }
}

type UpstreamByteStream =
    Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>;

struct AnthropicSseState {
    upstream: UpstreamByteStream,
    buffer: String,
    pending: VecDeque<Result<ProviderStreamEvent, ProviderError>>,
}

fn anthropic_sse_provider_stream(response: reqwest::Response) -> ProviderStream {
    let state = AnthropicSseState {
        upstream: Box::pin(response.bytes_stream()),
        buffer: String::new(),
        pending: VecDeque::new(),
    };

    Box::pin(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.pending.pop_front() {
                return Some((event, state));
            }

            if let Some(frame) = pop_sse_frame(&mut state.buffer) {
                state.pending.extend(parse_anthropic_sse_frame(&frame));
                continue;
            }

            match state.upstream.next().await {
                Some(Ok(bytes)) => {
                    state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Some(Err(error)) => {
                    return Some((Err(ProviderError::Rejected(error.to_string())), state));
                }
                None => {
                    if !state.buffer.trim().is_empty() {
                        let frame = std::mem::take(&mut state.buffer);
                        state.pending.extend(parse_anthropic_sse_frame(&frame));
                        continue;
                    }
                    return None;
                }
            }
        }
    }))
}

fn pop_sse_frame(buffer: &mut String) -> Option<String> {
    let index = buffer.find("\n\n")?;
    let frame = buffer[..index].to_string();
    buffer.drain(..index + 2);
    Some(frame)
}

fn parse_anthropic_sse_frame(frame: &str) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let event = frame
        .lines()
        .find_map(|line| line.strip_prefix("event:"))
        .map(str::trim);
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");

    if data.trim().is_empty() || event == Some("ping") {
        return Vec::new();
    }

    let value = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(value) => value,
        Err(error) => {
            return vec![Err(ProviderError::Rejected(format!(
                "invalid Anthropic stream event: {error}"
            )))];
        }
    };

    if event == Some("error") || value.get("type").and_then(|kind| kind.as_str()) == Some("error") {
        return vec![Err(ProviderError::Rejected(value.to_string()))];
    }

    anthropic_stream_events_from_event(event, value)
}

fn anthropic_stream_events_from_event(
    event: Option<&str>,
    value: serde_json::Value,
) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let event_type = value.get("type").and_then(|kind| kind.as_str()).or(event);
    match event_type {
        Some("content_block_start") => anthropic_content_block_start_event(value),
        Some("content_block_delta") => anthropic_content_block_delta_event(value),
        Some("message_delta") => anthropic_message_delta_stream_event(value),
        Some("message_stop") => Vec::new(),
        _ => Vec::new(),
    }
}

fn anthropic_content_block_start_event(
    value: serde_json::Value,
) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let Some(block) = value.get("content_block") else {
        return Vec::new();
    };
    if block.get("type").and_then(|kind| kind.as_str()) != Some("tool_use") {
        return Vec::new();
    }

    vec![Ok(ProviderStreamEvent::ToolCallDelta {
        index: value
            .get("index")
            .and_then(|index| index.as_u64())
            .unwrap_or(0) as usize,
        id: block
            .get("id")
            .and_then(|id| id.as_str())
            .map(ToolCallId::from),
        name: block
            .get("name")
            .and_then(|name| name.as_str())
            .map(str::to_string),
        arguments_delta: String::new(),
    })]
}

fn anthropic_content_block_delta_event(
    value: serde_json::Value,
) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let Some(delta) = value.get("delta") else {
        return Vec::new();
    };
    match delta.get("type").and_then(|kind| kind.as_str()) {
        Some("text_delta") => delta
            .get("text")
            .and_then(|text| text.as_str())
            .filter(|text| !text.is_empty())
            .map(|text| {
                vec![Ok(ProviderStreamEvent::TextDelta {
                    text: text.to_string(),
                })]
            })
            .unwrap_or_default(),
        Some("input_json_delta") => vec![Ok(ProviderStreamEvent::ToolCallDelta {
            index: value
                .get("index")
                .and_then(|index| index.as_u64())
                .unwrap_or(0) as usize,
            id: None,
            name: None,
            arguments_delta: delta
                .get("partial_json")
                .and_then(|json| json.as_str())
                .unwrap_or_default()
                .to_string(),
        })],
        Some("thinking_delta") | Some("signature_delta") => Vec::new(),
        _ => Vec::new(),
    }
}

fn anthropic_message_delta_stream_event(
    value: serde_json::Value,
) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let reason = value
        .get("delta")
        .and_then(|delta| delta.get("stop_reason"))
        .and_then(|reason| reason.as_str())
        .map(anthropic_provider_finish_reason)
        .unwrap_or(ProviderFinishReason::Stop);
    let mut events = Vec::new();

    if let Some(usage) = value.get("usage").and_then(|usage| {
        serde_json::from_value::<provider_core::ProviderUsage>(usage.clone()).ok()
    }) {
        events.push(Ok(ProviderStreamEvent::Usage { usage }));
    }
    events.push(Ok(ProviderStreamEvent::Finish { reason }));

    events
}

fn anthropic_provider_finish_reason(reason: &str) -> ProviderFinishReason {
    match reason {
        "end_turn" | "stop_sequence" => ProviderFinishReason::Stop,
        "tool_use" => ProviderFinishReason::ToolCalls,
        "max_tokens" => ProviderFinishReason::Length,
        other => ProviderFinishReason::Other(other.to_string()),
    }
}

fn anthropic_media_block(media: &MediaArtifact) -> serde_json::Value {
    match &media.source {
        MediaSource::Base64 { data } => serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media.media_type,
                "data": data
            }
        }),
        MediaSource::DataUrl { data_url } => {
            let (media_type, data) = split_data_url(data_url)
                .unwrap_or_else(|| (media.media_type.clone(), data_url.clone()));
            serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data
                }
            })
        }
        MediaSource::RemoteUrl { url } => serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": url
            }
        }),
    }
}

fn anthropic_messages(request: &ProviderRequest) -> Vec<serde_json::Value> {
    if request.messages.is_empty() || should_fallback_to_text_tool_results(request) {
        return vec![anthropic_fallback_user_message(request)];
    }

    let messages = request
        .messages
        .iter()
        .filter(|message| message.role != MessageRole::System)
        .filter_map(|message| anthropic_normalized_message(message, request))
        .collect::<Vec<_>>();

    if messages.is_empty() {
        vec![anthropic_fallback_user_message(request)]
    } else {
        messages
    }
}

fn anthropic_fallback_user_message(request: &ProviderRequest) -> serde_json::Value {
    let input_text = input_text_with_tool_results(&request.input_text, &request.tool_results);
    let mut content = vec![serde_json::json!({
        "type": "text",
        "text": input_text
    })];

    for media in &request.media_artifacts {
        content.push(anthropic_media_block(media));
    }

    serde_json::json!({
        "role": "user",
        "content": content
    })
}

fn should_fallback_to_text_tool_results(request: &ProviderRequest) -> bool {
    !request.tool_results.is_empty() && !messages_contain_tool_calls(&request.messages)
}

fn messages_contain_tool_calls(messages: &[NormalizedMessage]) -> bool {
    messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, NormalizedContentPart::ToolCall { .. }))
    })
}

fn anthropic_normalized_message(
    message: &NormalizedMessage,
    request: &ProviderRequest,
) -> Option<serde_json::Value> {
    let content = message
        .content
        .iter()
        .filter_map(|part| anthropic_content_block(part, request))
        .collect::<Vec<_>>();

    if content.is_empty() {
        return None;
    }

    Some(serde_json::json!({
        "role": anthropic_role(&message.role),
        "content": content
    }))
}

fn anthropic_content_block(
    part: &NormalizedContentPart,
    request: &ProviderRequest,
) -> Option<serde_json::Value> {
    match part {
        NormalizedContentPart::Text { text } => Some(serde_json::json!({
            "type": "text",
            "text": text
        })),
        NormalizedContentPart::Image { artifact_ref } => request
            .media_artifacts
            .iter()
            .find(|media| media.id == artifact_ref.artifact_id)
            .map(anthropic_media_block),
        NormalizedContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments_json,
        } => Some(serde_json::json!({
            "type": "tool_use",
            "id": tool_call_id.as_ref(),
            "name": tool_name,
            "input": arguments_json
        })),
        NormalizedContentPart::ToolResult { tool_call_id } => Some(serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_call_id.as_ref(),
            "content": tool_result_json(tool_call_id, request)
        })),
    }
}

fn anthropic_role(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::Assistant => "assistant",
        MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
    }
}

fn tool_result_json(tool_call_id: &ToolCallId, request: &ProviderRequest) -> serde_json::Value {
    request
        .tool_results
        .iter()
        .find(|result| &result.tool_call_id == tool_call_id)
        .map(|result| result.result_json.clone())
        .unwrap_or(serde_json::Value::Null)
}

fn anthropic_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    let mut value = serde_json::Map::new();
    value.insert(
        "name".to_string(),
        serde_json::Value::String(tool.name.clone()),
    );
    if let Some(description) = &tool.description {
        value.insert(
            "description".to_string(),
            serde_json::Value::String(description.clone()),
        );
    }
    value.insert("input_schema".to_string(), tool.input_schema.clone());
    serde_json::Value::Object(value)
}

fn anthropic_tool_choice(
    choice: &ToolChoice,
    parallel_tool_calls: Option<bool>,
    thinking_enabled: bool,
) -> Option<serde_json::Value> {
    let effective_choice =
        if thinking_enabled && matches!(choice, ToolChoice::Required | ToolChoice::Named { .. }) {
            ToolChoice::Auto
        } else {
            choice.clone()
        };

    let mut value = match effective_choice {
        ToolChoice::Auto if parallel_tool_calls.is_none() => return None,
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::Named { name } => serde_json::json!({
            "type": "tool",
            "name": name
        }),
    };

    if let Some(false) = parallel_tool_calls {
        value["disable_parallel_tool_use"] = serde_json::Value::Bool(true);
    }
    Some(value)
}

fn input_text_with_tool_results(input_text: &str, results: &[ToolResultRecord]) -> String {
    if results.is_empty() {
        return input_text.to_string();
    }

    let result_lines = results
        .iter()
        .map(|result| {
            format!(
                "- tool_call_id={} status={:?} result={}",
                result.tool_call_id.as_ref(),
                result.status,
                result.result_json
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    if input_text.trim().is_empty() {
        format!("Tool results available to the main agent:\n{result_lines}")
    } else {
        format!("{input_text}\n\nTool results available to the main agent:\n{result_lines}")
    }
}

fn parse_structured_or_text(
    scope: &IsolationKey,
    task_id: &TaskId,
    content: &str,
) -> Option<AgentArtifact> {
    let spawn_plan = serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .filter(|value| value.get("type").and_then(|kind| kind.as_str()) == Some("spawn_plan"))
        .and_then(|value| value.get("plan").cloned())
        .and_then(|plan_value| serde_json::from_value::<SpawnPlan>(plan_value).ok());

    if let Some(plan) = spawn_plan {
        return Some(AgentArtifact::SpawnPlan {
            id: ArtifactId::from(format!("spawn-plan-{}", task_id.as_ref())),
            scope: scope.clone(),
            plan,
        });
    }

    Some(AgentArtifact::Text {
        id: ArtifactId::from(format!("text-{}", task_id.as_ref())),
        scope: scope.clone(),
        text: content.to_string(),
    })
}

fn split_data_url(data_url: &str) -> Option<(String, String)> {
    let rest = data_url.strip_prefix("data:")?;
    let (media_type, data) = rest.split_once(";base64,")?;
    Some((media_type.to_string(), data.to_string()))
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
    use provider_core::ProviderRequest;

    #[test]
    fn anthropic_payload_contains_multimodal_content() {
        let request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["model"], "claude-test");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["messages"][0]["content"][1]["type"], "image");
        assert_eq!(
            body["messages"][0]["content"][1]["source"]["media_type"],
            "image/png"
        );
        assert_eq!(body["messages"][0]["content"][1]["source"]["data"], "AAAA");
    }

    #[test]
    fn anthropic_payload_preserves_remote_image_url() {
        let request = provider_request_with_media(MediaSource::RemoteUrl {
            url: "https://example.com/image.png".to_string(),
        });

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["messages"][0]["content"][1]["type"], "image");
        assert_eq!(body["messages"][0]["content"][1]["source"]["type"], "url");
        assert_eq!(
            body["messages"][0]["content"][1]["source"]["url"],
            "https://example.com/image.png"
        );
    }

    #[test]
    fn anthropic_response_parses_tool_use() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = AnthropicProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "content": [{
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "lookup",
                    "input": {"q": "rust"}
                }]
            }),
        )
        .unwrap();

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool_name, "lookup");
        assert_eq!(response.tool_calls[0].arguments_json["q"], "rust");
    }

    #[test]
    fn anthropic_response_preserves_usage() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = AnthropicProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "content": [{"type": "text", "text": "Final answer"}],
                "usage": {
                    "input_tokens": 13,
                    "output_tokens": 8
                }
            }),
        )
        .unwrap();

        assert_eq!(response.usage.input_tokens, 13);
        assert_eq!(response.usage.output_tokens, 8);
    }

    #[test]
    fn anthropic_payload_exposes_tools_to_main_agent() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: Some("Look up weather by city".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        }];

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(
            body["tools"][0]["name"],
            serde_json::Value::String("lookup_weather".to_string())
        );
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn anthropic_payload_maps_required_and_parallel_tool_choice() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.tool_choice = ToolChoice::Required;
        request.parallel_tool_calls = Some(false);

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["tool_choice"]["type"], "any");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn anthropic_payload_forces_named_tool_when_requested() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.tool_choice = ToolChoice::Named {
            name: "lookup_weather".to_string(),
        };

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "lookup_weather");
    }

    #[test]
    fn anthropic_payload_preserves_model_provider_options() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.provider_options = serde_json::json!({
            "max_tokens": 777,
            "temperature": 0.8,
            "top_p": 0.5,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "user-a"}
        });

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["max_tokens"], 777);
        assert_eq!(body["temperature"], 0.8);
        assert_eq!(body["top_p"], 0.5);
        assert_eq!(body["stop_sequences"], serde_json::json!(["END"]));
        assert_eq!(body["metadata"], serde_json::json!({"user_id": "user-a"}));
    }

    #[test]
    fn anthropic_payload_feeds_tool_results_back_and_keeps_tools_available() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope: request.scope.clone(),
            result_json: serde_json::json!({"temperature": "21C"}),
            result_sha256: "hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["tools"][0]["name"], "lookup_weather");
        assert!(body.get("tool_choice").is_none());
        let text = body["messages"][0]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Tool results available to the main agent"));
        assert!(text.contains("call-1"));
        assert!(text.contains("21C"));
    }

    #[test]
    fn anthropic_payload_preserves_structured_tool_use_history() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.media_artifacts.clear();
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.messages = vec![
            NormalizedMessage {
                role: MessageRole::User,
                content: vec![NormalizedContentPart::Text {
                    text: "weather".to_string(),
                }],
            },
            NormalizedMessage {
                role: MessageRole::Assistant,
                content: vec![NormalizedContentPart::ToolCall {
                    tool_call_id: ToolCallId::from("call-1"),
                    tool_name: "lookup_weather".to_string(),
                    arguments_json: serde_json::json!({"city": "Taipei"}),
                }],
            },
            NormalizedMessage {
                role: MessageRole::User,
                content: vec![NormalizedContentPart::ToolResult {
                    tool_call_id: ToolCallId::from("call-1"),
                }],
            },
        ];
        request.tool_results = vec![ToolResultRecord {
            tool_call_id: ToolCallId::from("call-1"),
            scope: request.scope.clone(),
            result_json: serde_json::json!({"temperature": "21C"}),
            result_sha256: "hash".to_string(),
            status: ToolResultStatus::Accepted,
        }];

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["tools"][0]["name"], "lookup_weather");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][0]["name"], "lookup_weather");
        assert_eq!(body["messages"][2]["role"], "user");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call-1");
    }

    #[test]
    fn anthropic_payload_includes_user_system_instructions() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.system_instructions = vec!["Answer in Traditional Chinese.".to_string()];

        let body = AnthropicProvider::build_request_body(&request);

        assert!(
            body["system"]
                .as_str()
                .unwrap()
                .contains("Answer in Traditional Chinese.")
        );
    }

    #[test]
    fn anthropic_payload_enables_thinking_when_requested() {
        let mut request = provider_request_with_media(MediaSource::Base64 {
            data: "AAAA".to_string(),
        });
        request.thinking_enabled = true;

        let body = AnthropicProvider::build_request_body(&request);

        assert_eq!(body["thinking"]["type"], "enabled");
    }

    #[test]
    fn anthropic_stream_event_parses_text_delta() {
        let events = anthropic_stream_events_from_event(
            Some("content_block_delta"),
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"}
            }),
        )
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert_eq!(
            events,
            vec![ProviderStreamEvent::TextDelta {
                text: "hello".to_string()
            }]
        );
    }

    #[test]
    fn anthropic_stream_event_parses_tool_use_start_and_input_delta() {
        let start = anthropic_stream_events_from_event(
            Some("content_block_start"),
            serde_json::json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {}
                }
            }),
        )
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        let delta = anthropic_stream_events_from_event(
            Some("content_block_delta"),
            serde_json::json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"city\""}
            }),
        )
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert_eq!(
            start,
            vec![ProviderStreamEvent::ToolCallDelta {
                index: 1,
                id: Some(ToolCallId::from("toolu_1")),
                name: Some("lookup".to_string()),
                arguments_delta: String::new(),
            }]
        );
        assert_eq!(
            delta,
            vec![ProviderStreamEvent::ToolCallDelta {
                index: 1,
                id: None,
                name: None,
                arguments_delta: "{\"city\"".to_string(),
            }]
        );
    }

    fn provider_request_with_media(source: MediaSource) -> ProviderRequest {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        ProviderRequest {
            scope: scope.clone(),
            task: SubtaskSpec {
                task_id: TaskId::from("root"),
                parent_task_id: None,
                spawn_depth: 0,
                role: AgentRole::Leader,
                objective: "answer".to_string(),
                input_artifact_refs: vec![],
                expected_outputs: vec![ArtifactKind::Text],
                allowed_capabilities: CapabilitySet::from([Capability::Text, Capability::Image]),
                limits: AgentLimits::default(),
            },
            model: "claude-test".to_string(),
            system_instructions: vec![],
            thinking_enabled: false,
            thinking_format: ThinkingFormat::Auto,
            input_text: "inspect image".to_string(),
            messages: Vec::new(),
            media_artifacts: vec![MediaArtifact {
                id: ArtifactId::from("media-0"),
                scope,
                media_type: "image/png".to_string(),
                source,
                sha256: "hash".to_string(),
                byte_len: Some(4),
            }],
            artifacts: vec![],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            tool_results: vec![],
            provider_options: serde_json::json!({}),
        }
    }
}
