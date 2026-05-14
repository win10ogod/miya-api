use std::{collections::VecDeque, pin::Pin};

use agent_protocol::*;
use async_trait::async_trait;
use futures::StreamExt;
use provider_core::{
    ModelProvider, ProviderError, ProviderFinishReason, ProviderRequest, ProviderResponse,
    ProviderStream, ProviderStreamEvent,
};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ProviderError::Rejected("OPENAI_API_KEY is required".to_string()))?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        Ok(Self::new(base_url, api_key))
    }

    pub fn build_request_body(request: &ProviderRequest) -> serde_json::Value {
        let messages = openai_messages(request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages
        });
        merge_provider_options(&mut body, &request.provider_options);

        match effective_thinking_format(request) {
            ThinkingFormat::QwenDashScope => {
                body["enable_thinking"] = serde_json::Value::Bool(request.thinking_enabled);
                body["preserve_thinking"] = serde_json::Value::Bool(request.thinking_enabled);
            }
            ThinkingFormat::QwenChatTemplate | ThinkingFormat::Auto => {
                body["chat_template_kwargs"] = serde_json::json!({
                    "enable_thinking": request.thinking_enabled,
                    "preserve_thinking": request.thinking_enabled
                });
            }
            ThinkingFormat::GemmaSystemToken => {}
        }

        if request.tool_results.is_empty() && !request.tools.is_empty() {
            body["tools"] = serde_json::Value::Array(
                openai_tools_for_request(request)
                    .into_iter()
                    .map(openai_tool_definition)
                    .collect(),
            );
            body["tool_choice"] = openai_tool_choice_for_request(request);
            if let Some(parallel_tool_calls) = request.parallel_tool_calls {
                body["parallel_tool_calls"] = serde_json::Value::Bool(parallel_tool_calls);
            }
        }

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
        let message = value
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .ok_or_else(|| {
                ProviderError::Rejected("OpenAI response missing message".to_string())
            })?;

        let mut response = ProviderResponse::default();

        if let Some(tool_calls) = message.get("tool_calls").and_then(|calls| calls.as_array()) {
            for call in tool_calls {
                let id = call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("tool-call");
                let function = call.get("function").unwrap_or(&serde_json::Value::Null);
                let name = function
                    .get("name")
                    .and_then(|name| name.as_str())
                    .unwrap_or("unknown");
                let arguments = function
                    .get("arguments")
                    .and_then(|arguments| arguments.as_str())
                    .and_then(|arguments| serde_json::from_str(arguments).ok())
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
        }

        if let Some(artifact) = message
            .get("content")
            .and_then(|content| content.as_str())
            .filter(|content| !strip_thinking_markup(content).trim().is_empty())
            .and_then(|content| parse_structured_or_text(scope, task_id, content))
        {
            response.artifacts.push(artifact);
        } else if let Some(artifact) = message
            .get("reasoning")
            .and_then(|reasoning| reasoning.as_str())
            .and_then(extract_final_answer_from_reasoning)
            .and_then(|content| parse_structured_or_text(scope, task_id, &content))
        {
            response.artifacts.push(artifact);
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

fn format_system_prompt(
    role: &AgentRole,
    objective: &str,
    user_system_instructions: &[String],
    thinking_enabled: bool,
    thinking_format: &ThinkingFormat,
) -> String {
    let user_instructions = if user_system_instructions.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nUser/system instructions to preserve:\n{}",
            user_system_instructions.join("\n")
        )
    };
    let prompt = format!(
        "You are a bounded multi-agent worker. Role: {role:?}. Objective: {objective}. If proposing child agents, return JSON with type=spawn_plan and a plan object. Final answers must be natural, directly useful, and must not expose sub-agent state, internal tool calls, or orchestration details. Preserve formatting exactly when the answer contains XML/HTML-like tags, Markdown, fenced code, lists, tables, or delimiter-separated blocks. Do not minify, collapse line breaks, remove spaces around headings, or merge tag-delimited sections.{user_instructions}"
    );
    if thinking_enabled && thinking_format == &ThinkingFormat::GemmaSystemToken {
        format!("<|think|>\n{prompt}")
    } else {
        prompt
    }
}

fn effective_thinking_format(request: &ProviderRequest) -> ThinkingFormat {
    if request.thinking_format == ThinkingFormat::Auto {
        if is_gemma_generation_model(&request.model) {
            ThinkingFormat::GemmaSystemToken
        } else {
            ThinkingFormat::QwenChatTemplate
        }
    } else {
        request.thinking_format.clone()
    }
}

fn merge_provider_options(body: &mut serde_json::Value, options: &serde_json::Value) {
    let (Some(body), Some(options)) = (body.as_object_mut(), options.as_object()) else {
        return;
    };
    for (key, value) in options {
        if !value.is_null() && !is_openai_core_field(key) {
            body.insert(key.clone(), value.clone());
        }
    }
}

fn is_openai_core_field(key: &str) -> bool {
    matches!(
        key,
        "model"
            | "messages"
            | "tools"
            | "tool_choice"
            | "parallel_tool_calls"
            | "stream"
            | "functions"
            | "function_call"
    )
}

fn is_gemma_generation_model(model: &str) -> bool {
    let configured_lists = [
        "MIYA_GEMMA_MODELS",
        "MULTI_AGENT_GEMMA_MODELS",
        "GEMMA_MODELS",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok());
    is_gemma_generation_model_with_config(model, configured_lists)
}

fn is_gemma_generation_model_with_config<I, S>(model: &str, configured_lists: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let normalized = normalize_model_id(model);
    if normalized.is_empty() {
        return false;
    }
    normalized.contains("gemma")
        || configured_lists
            .into_iter()
            .any(|raw| configured_model_list_contains(raw.as_ref(), &normalized))
}

fn configured_model_list_contains(raw: &str, normalized_model: &str) -> bool {
    raw.split(',')
        .map(normalize_model_id)
        .any(|configured| configured == normalized_model)
}

fn normalize_model_id(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let scope = request.scope.clone();
        let task_id = request.task.task_id.clone();
        let body = Self::build_request_body(&request);
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?;
        let value = json_or_error(response, "OpenAI chat completion")
            .await?
            .json::<serde_json::Value>()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?;

        Self::parse_response(&scope, &task_id, value)
    }

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream, ProviderError> {
        let body = Self::build_stream_request_body(&request);
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderError::Rejected(error.to_string()))?;
        let response = json_or_error(response, "OpenAI chat completion stream").await?;

        Ok(openai_sse_provider_stream(response))
    }
}

async fn json_or_error(
    response: reqwest::Response,
    context: &str,
) -> Result<reqwest::Response, ProviderError> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let url = response.url().to_string();
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<failed to read error body: {error}>"));
    Err(ProviderError::Rejected(format!(
        "{context} failed: HTTP {status} for {url}; body: {body}"
    )))
}

type UpstreamByteStream =
    Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>;

struct OpenAiSseState {
    upstream: UpstreamByteStream,
    buffer: String,
    pending: VecDeque<Result<ProviderStreamEvent, ProviderError>>,
}

fn openai_sse_provider_stream(response: reqwest::Response) -> ProviderStream {
    let state = OpenAiSseState {
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
                state.pending.extend(parse_openai_sse_frame(&frame));
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
                        state.pending.extend(parse_openai_sse_frame(&frame));
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

fn parse_openai_sse_frame(frame: &str) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.trim().is_empty() {
        return Vec::new();
    }
    if data.trim() == "[DONE]" {
        return Vec::new();
    }

    let value = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(value) => value,
        Err(error) => {
            return vec![Err(ProviderError::Rejected(format!(
                "invalid OpenAI stream chunk: {error}"
            )))];
        }
    };
    if let Some(error) = value.get("error") {
        return vec![Err(ProviderError::Rejected(error.to_string()))];
    }

    openai_stream_events_from_chunk(value)
}

fn openai_stream_events_from_chunk(
    value: serde_json::Value,
) -> Vec<Result<ProviderStreamEvent, ProviderError>> {
    let mut events = Vec::new();

    if let Some(usage) = value.get("usage").and_then(|usage| {
        serde_json::from_value::<provider_core::ProviderUsage>(usage.clone()).ok()
    }) {
        events.push(Ok(ProviderStreamEvent::Usage { usage }));
    }

    let Some(choice) = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
    else {
        return events;
    };

    if let Some(delta) = choice.get("delta") {
        if let Some(text) = delta.get("content").and_then(|text| text.as_str())
            && !text.is_empty()
        {
            events.push(Ok(ProviderStreamEvent::TextDelta {
                text: strip_thinking_markup(text),
            }));
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|calls| calls.as_array()) {
            for call in tool_calls {
                let function = call.get("function").unwrap_or(&serde_json::Value::Null);
                events.push(Ok(ProviderStreamEvent::ToolCallDelta {
                    index: call
                        .get("index")
                        .and_then(|index| index.as_u64())
                        .unwrap_or(0) as usize,
                    id: call
                        .get("id")
                        .and_then(|id| id.as_str())
                        .map(ToolCallId::from),
                    name: function
                        .get("name")
                        .and_then(|name| name.as_str())
                        .map(str::to_string),
                    arguments_delta: function
                        .get("arguments")
                        .and_then(|arguments| arguments.as_str())
                        .unwrap_or_default()
                        .to_string(),
                }));
            }
        }

        if let Some(function_call) = delta.get("function_call") {
            events.push(Ok(ProviderStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: function_call
                    .get("name")
                    .and_then(|name| name.as_str())
                    .map(str::to_string),
                arguments_delta: function_call
                    .get("arguments")
                    .and_then(|arguments| arguments.as_str())
                    .unwrap_or_default()
                    .to_string(),
            }));
        }
    }

    if let Some(reason) = choice
        .get("finish_reason")
        .and_then(|reason| reason.as_str())
    {
        events.push(Ok(ProviderStreamEvent::Finish {
            reason: openai_provider_finish_reason(reason),
        }));
    }

    events
}

fn openai_provider_finish_reason(reason: &str) -> ProviderFinishReason {
    match reason {
        "stop" => ProviderFinishReason::Stop,
        "tool_calls" => ProviderFinishReason::ToolCalls,
        "function_call" => ProviderFinishReason::FunctionCall,
        "length" => ProviderFinishReason::Length,
        other => ProviderFinishReason::Other(other.to_string()),
    }
}

fn openai_image_url(media: &MediaArtifact) -> String {
    match &media.source {
        MediaSource::DataUrl { data_url } => data_url.clone(),
        MediaSource::RemoteUrl { url } => url.clone(),
        MediaSource::Base64 { data } => format!("data:{};base64,{}", media.media_type, data),
    }
}

fn openai_messages(request: &ProviderRequest) -> Vec<serde_json::Value> {
    let mut messages = vec![serde_json::json!({
        "role": "system",
        "content": format_system_prompt(
            &request.task.role,
            &request.task.objective,
            &request.system_instructions,
            request.thinking_enabled,
            &effective_thinking_format(request)
        )
    })];

    if request.messages.is_empty() || should_fallback_to_text_tool_results(request) {
        messages.push(openai_fallback_user_message(request));
        return messages;
    }

    for message in &request.messages {
        if message.role == MessageRole::System {
            continue;
        }
        messages.extend(openai_normalized_message(message, request));
    }

    if messages.len() == 1 {
        messages.push(openai_fallback_user_message(request));
    }

    messages
}

fn openai_fallback_user_message(request: &ProviderRequest) -> serde_json::Value {
    let input_text = input_text_with_tool_results(&request.input_text, &request.tool_results);
    let mut content = vec![serde_json::json!({
        "type": "text",
        "text": input_text
    })];

    for media in &request.media_artifacts {
        content.push(serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": openai_image_url(media)
            }
        }));
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

fn openai_normalized_message(
    message: &NormalizedMessage,
    request: &ProviderRequest,
) -> Vec<serde_json::Value> {
    let tool_results = message
        .content
        .iter()
        .filter_map(|part| match part {
            NormalizedContentPart::ToolResult { tool_call_id } => {
                Some(openai_tool_result_message(tool_call_id, request))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if !tool_results.is_empty() {
        return tool_results;
    }

    match message.role {
        MessageRole::Assistant => vec![openai_assistant_message(message)],
        MessageRole::Tool => Vec::new(),
        MessageRole::User | MessageRole::System => vec![serde_json::json!({
            "role": openai_role(&message.role),
            "content": openai_content_blocks(&message.content, request)
        })],
    }
}

fn openai_assistant_message(message: &NormalizedMessage) -> serde_json::Value {
    let text = message_text_parts(&message.content).join("\n");
    let tool_calls = message
        .content
        .iter()
        .filter_map(|part| match part {
            NormalizedContentPart::ToolCall {
                tool_call_id,
                tool_name,
                arguments_json,
            } => Some(serde_json::json!({
                "id": tool_call_id.as_ref(),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments_json.to_string()
                }
            })),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut value = serde_json::json!({
        "role": "assistant",
        "content": if text.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(text)
        }
    });
    if !tool_calls.is_empty() {
        value["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    value
}

fn openai_tool_result_message(
    tool_call_id: &ToolCallId,
    request: &ProviderRequest,
) -> serde_json::Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": tool_call_id.as_ref(),
        "content": tool_result_json(tool_call_id, request).to_string()
    })
}

fn openai_content_blocks(
    parts: &[NormalizedContentPart],
    request: &ProviderRequest,
) -> serde_json::Value {
    let content = parts
        .iter()
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(serde_json::json!({
                "type": "text",
                "text": text
            })),
            NormalizedContentPart::Image { artifact_ref } => request
                .media_artifacts
                .iter()
                .find(|media| media.id == artifact_ref.artifact_id)
                .map(|media| {
                    serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": openai_image_url(media)
                        }
                    })
                }),
            NormalizedContentPart::ToolCall { .. } | NormalizedContentPart::ToolResult { .. } => {
                None
            }
        })
        .collect::<Vec<_>>();

    serde_json::Value::Array(content)
}

fn openai_role(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn message_text_parts(parts: &[NormalizedContentPart]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|part| match part {
            NormalizedContentPart::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_json(tool_call_id: &ToolCallId, request: &ProviderRequest) -> serde_json::Value {
    request
        .tool_results
        .iter()
        .find(|result| &result.tool_call_id == tool_call_id)
        .map(|result| result.result_json.clone())
        .unwrap_or(serde_json::Value::Null)
}

fn openai_tool_definition(tool: &ToolDefinition) -> serde_json::Value {
    let mut function = serde_json::Map::new();
    function.insert(
        "name".to_string(),
        serde_json::Value::String(tool.name.clone()),
    );
    if let Some(description) = &tool.description {
        function.insert(
            "description".to_string(),
            serde_json::Value::String(description.clone()),
        );
    }
    function.insert(
        "parameters".to_string(),
        normalized_openai_tool_parameters(&tool.input_schema),
    );

    serde_json::json!({
        "type": "function",
        "function": function
    })
}

fn normalized_openai_tool_parameters(input_schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = input_schema.clone();
    if !schema.is_object() {
        schema = serde_json::json!({});
    }

    let object = schema.as_object_mut().expect("schema object");
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

    schema
}

fn openai_tools_for_request(request: &ProviderRequest) -> Vec<&ToolDefinition> {
    if should_adapt_named_tool_choice_for_local_backend(request)
        && let ToolChoice::Named { name } = &request.tool_choice
    {
        return request
            .tools
            .iter()
            .filter(|tool| tool.name == *name)
            .collect();
    }

    request.tools.iter().collect()
}

fn openai_tool_choice_for_request(request: &ProviderRequest) -> serde_json::Value {
    if should_adapt_named_tool_choice_for_local_backend(request) {
        return serde_json::Value::String("required".to_string());
    }

    openai_tool_choice(&request.tool_choice)
}

fn should_adapt_named_tool_choice_for_local_backend(request: &ProviderRequest) -> bool {
    is_gemma_generation_model(&request.model)
        && matches!(request.tool_choice, ToolChoice::Named { .. })
}

fn openai_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::Value::String("auto".to_string()),
        ToolChoice::None => serde_json::Value::String("none".to_string()),
        ToolChoice::Required => serde_json::Value::String("required".to_string()),
        ToolChoice::Named { name } => serde_json::json!({
            "type": "function",
            "function": {
                "name": name
            }
        }),
    }
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
        text: strip_thinking_markup(content),
    })
}

fn strip_thinking_markup(content: &str) -> String {
    if let Some((_, after)) = content.rsplit_once("<channel|>") {
        return strip_generation_wrappers(after);
    }
    if let Some((_, after)) = content.rsplit_once("</think>") {
        return strip_generation_wrappers(after);
    }
    strip_generation_wrappers(content)
}

fn strip_generation_wrappers(content: &str) -> String {
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

fn extract_final_answer_from_reasoning(reasoning: &str) -> Option<String> {
    let marked = after_last_final_marker(reasoning)?;

    for (open, close) in [('「', '」'), ('“', '”'), ('"', '"'), ('`', '`')] {
        if let Some(candidate) = last_enclosed(marked, open, close).filter(|candidate| {
            is_public_answer_candidate(candidate) && !looks_like_reasoning(candidate)
        }) {
            return Some(candidate);
        }
    }

    marked
        .lines()
        .rev()
        .map(clean_final_answer_line)
        .find(|candidate| is_public_answer_candidate(candidate) && !looks_like_reasoning(candidate))
}

fn after_last_final_marker(reasoning: &str) -> Option<&str> {
    let markers = [
        "Final answer:",
        "Final Answer:",
        "final answer:",
        "Final output:",
        "Final Output:",
        "Final Output Generation",
        "Final Polish",
        "Final:",
        "final:",
        "最終答案：",
        "最终答案：",
        "最終回答：",
        "最终回答：",
        "最終輸出：",
        "最终输出：",
        "最終回覆：",
        "最终回复：",
        "答案：",
        "回答：",
    ];

    markers
        .iter()
        .filter_map(|marker| reasoning.rfind(marker).map(|index| (index, marker.len())))
        .max_by_key(|(index, _)| *index)
        .map(|(index, len)| &reasoning[index + len..])
}

fn last_enclosed(content: &str, open: char, close: char) -> Option<String> {
    let close_index = content.rfind(close)?;
    let before_close = &content[..close_index];
    let open_index = before_close.rfind(open)?;
    Some(
        before_close[open_index + open.len_utf8()..]
            .trim()
            .to_string(),
    )
}

fn clean_final_answer_line(line: &str) -> String {
    line.trim()
        .trim_start_matches(['-', '*', ' ', '\t'])
        .trim_start_matches("Final answer:")
        .trim_start_matches("Final Answer:")
        .trim_start_matches("final answer:")
        .trim_start_matches("Final:")
        .trim_start_matches("final:")
        .trim_start_matches("最終答案：")
        .trim_start_matches("最终答案：")
        .trim_start_matches("最終回答：")
        .trim_start_matches("最终回答：")
        .trim_start_matches("答案：")
        .trim_start_matches("回答：")
        .trim()
        .to_string()
}

fn is_public_answer_candidate(candidate: &str) -> bool {
    let trimmed = candidate.trim();
    !trimmed.is_empty()
        && trimmed.chars().count() >= 4
        && trimmed.lines().count() <= 6
        && trimmed.chars().count() <= 1200
}

fn looks_like_reasoning(candidate: &str) -> bool {
    let lower = candidate.to_lowercase();
    [
        "analyze",
        "reasoning",
        "thought",
        "step by step",
        "private",
        "思考",
        "推理",
        "分析",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
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
    fn openai_payload_contains_multimodal_content() {
        let request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "text");
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn openai_response_parses_tool_calls() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\":\"rust\"}"
                            }
                        }]
                    }
                }]
            }),
        )
        .unwrap();

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool_name, "lookup");
        assert_eq!(response.tool_calls[0].arguments_json["q"], "rust");
    }

    #[test]
    fn openai_payload_exposes_function_tools_to_main_agent() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
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

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(
            body["tools"][0]["function"]["name"],
            serde_json::Value::String("lookup_weather".to_string())
        );
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn openai_payload_forces_named_tool_when_requested() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.tool_choice = ToolChoice::Named {
            name: "lookup_weather".to_string(),
        };
        request.parallel_tool_calls = Some(false);

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "lookup_weather");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn openai_payload_preserves_model_provider_options() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.provider_options = serde_json::json!({
            "temperature": 0.9,
            "top_p": 0.4,
            "max_completion_tokens": 321,
            "response_format": {"type": "json_object"},
            "seed": 42,
            "metadata": {"foo": "bar"}
        });

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["top_p"], 0.4);
        assert_eq!(body["max_completion_tokens"], 321);
        assert_eq!(
            body["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        assert_eq!(body["seed"], 42);
        assert_eq!(body["metadata"], serde_json::json!({"foo": "bar"}));
    }

    #[test]
    fn openai_payload_does_not_force_sampling_when_unconfigured() {
        let request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });

        let body = OpenAiProvider::build_request_body(&request);

        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn openai_payload_adapts_named_tool_choice_for_gemma_backend() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.model = "local-gemma-finetune".to_string();
        request.tools = vec![
            ToolDefinition {
                name: "lookup_weather".to_string(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "lookup_news".to_string(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            },
        ];
        request.tool_choice = ToolChoice::Named {
            name: "lookup_weather".to_string(),
        };

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            body["tools"][0]["function"]["name"],
            serde_json::Value::String("lookup_weather".to_string())
        );
    }

    #[test]
    fn openai_payload_can_disable_tool_use_with_tool_choice_none() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];
        request.tool_choice = ToolChoice::None;

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["tool_choice"], "none");
        assert!(body["tools"].is_array());
    }

    #[test]
    fn openai_payload_adds_missing_tool_parameter_properties_for_local_backend() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.tools = vec![ToolDefinition {
            name: "lookup_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        }];

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn openai_payload_feeds_tool_results_back_without_recalling_tools() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
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

        let body = OpenAiProvider::build_request_body(&request);

        assert!(body.get("tools").is_none());
        let text = body["messages"][1]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Tool results available to the main agent"));
        assert!(text.contains("call-1"));
        assert!(text.contains("21C"));
    }

    #[test]
    fn openai_payload_preserves_structured_tool_call_history() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.media_artifacts.clear();
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
                role: MessageRole::Tool,
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

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["name"],
            "lookup_weather"
        );
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "call-1");
        assert!(
            body["messages"][3]["content"]
                .as_str()
                .unwrap()
                .contains("21C")
        );
    }

    #[test]
    fn openai_payload_includes_user_system_instructions() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.system_instructions = vec!["Answer in Traditional Chinese.".to_string()];

        let body = OpenAiProvider::build_request_body(&request);

        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("Answer in Traditional Chinese.")
        );
    }

    #[test]
    fn openai_payload_instructs_model_to_preserve_structured_formatting() {
        let request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });

        let body = OpenAiProvider::build_request_body(&request);
        let system = body["messages"][0]["content"].as_str().unwrap();

        assert!(system.contains("Preserve formatting exactly"));
        assert!(system.contains("Do not minify"));
        assert!(system.contains("XML/HTML-like tags"));
        assert!(system.contains("Markdown"));
    }

    #[test]
    fn openai_payload_disables_qwen_thinking_when_supported() {
        let request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn openai_payload_enables_qwen_thinking_when_requested() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.thinking_enabled = true;

        let body = OpenAiProvider::build_request_body(&request);

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn openai_payload_uses_gemma_system_token_format() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.model = "google/gemma-4-31B-it".to_string();
        request.thinking_enabled = true;
        request.thinking_format = ThinkingFormat::GemmaSystemToken;

        let body = OpenAiProvider::build_request_body(&request);

        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("<|think|>")
        );
    }

    #[test]
    fn openai_payload_uses_gemma_format_for_gemma_named_model() {
        let mut request = provider_request_with_media(MediaSource::DataUrl {
            data_url: "data:image/png;base64,AAAA".to_string(),
        });
        request.model = "local-gemma-finetune".to_string();
        request.thinking_enabled = true;
        request.thinking_format = ThinkingFormat::Auto;

        let body = OpenAiProvider::build_request_body(&request);

        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("<|think|>")
        );
        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn openai_response_strips_gemma_thought_channel() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "<|channel>thought\nprivate reasoning<channel|>\nFinal answer"
                    }
                }]
            }),
        )
        .unwrap();

        let AgentArtifact::Text { text, .. } = &response.artifacts[0] else {
            panic!("expected text artifact");
        };
        assert_eq!(text, "Final answer");
    }

    #[test]
    fn openai_response_strips_gemma_generation_wrappers() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "<bos><start_of_turn>model\n可直接給使用者的答案。<end_of_turn><eos>"
                    }
                }]
            }),
        )
        .unwrap();

        let AgentArtifact::Text { text, .. } = &response.artifacts[0] else {
            panic!("expected text artifact");
        };
        assert_eq!(text, "可直接給使用者的答案。");
    }

    #[test]
    fn gemma_generation_model_can_be_selected_by_configuration() {
        assert!(is_gemma_generation_model_with_config(
            "configured-local-alias",
            ["configured-local-alias"]
        ));
        assert!(!is_gemma_generation_model_with_config(
            "configured-local-alias",
            ["other-model"]
        ));
    }

    #[test]
    fn openai_response_maps_prompt_and_completion_usage() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {"content": "Final answer"}
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18
                }
            }),
        )
        .unwrap();

        assert_eq!(response.usage.input_tokens, 11);
        assert_eq!(response.usage.output_tokens, 7);
    }

    #[test]
    fn openai_response_uses_final_answer_from_reasoning_without_leaking_reasoning() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": null,
                        "reasoning": "Analyze the user request privately. Final answer: 「系統提示生效：這是可直接給使用者的答案。」"
                    }
                }]
            }),
        )
        .unwrap();

        let AgentArtifact::Text { text, .. } = &response.artifacts[0] else {
            panic!("expected text artifact");
        };
        assert_eq!(text, "系統提示生效：這是可直接給使用者的答案。");
        assert!(!text.contains("Analyze the user request"));
        assert!(!text.contains("reasoning"));
    }

    #[test]
    fn openai_response_does_not_expose_reasoning_without_final_marker() {
        let scope = IsolationKey::new("tenant", "request", "conversation");
        let task_id = TaskId::from("root");
        let response = OpenAiProvider::parse_response(
            &scope,
            &task_id,
            serde_json::json!({
                "choices": [{
                    "message": {
                        "content": null,
                        "reasoning": "Analyze privately. The answer might be hidden here but there is no public final marker."
                    }
                }]
            }),
        )
        .unwrap();

        assert!(response.artifacts.is_empty());
    }

    #[test]
    fn openai_stream_chunk_parses_text_delta() {
        let events = openai_stream_events_from_chunk(serde_json::json!({
            "choices": [{
                "delta": {"content": "hello"},
                "finish_reason": null
            }]
        }))
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
    fn openai_stream_chunk_parses_tool_call_delta_and_finish() {
        let events = openai_stream_events_from_chunk(serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\""}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert_eq!(
            events,
            vec![
                ProviderStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some(ToolCallId::from("call-1")),
                    name: Some("lookup".to_string()),
                    arguments_delta: "{\"city\"".to_string(),
                },
                ProviderStreamEvent::Finish {
                    reason: ProviderFinishReason::ToolCalls
                }
            ]
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
            model: "gpt-test".to_string(),
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
