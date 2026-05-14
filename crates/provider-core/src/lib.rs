use std::{pin::Pin, sync::Arc};

use agent_protocol::{
    AgentArtifact, IsolationKey, MediaArtifact, NormalizedMessage, SubtaskSpec, ToolCallId,
    ToolCallRecord, ToolChoice, ToolDefinition, ToolResultRecord,
};
use async_trait::async_trait;
use futures::{Stream, stream};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub scope: IsolationKey,
    pub task: SubtaskSpec,
    pub model: String,
    pub system_instructions: Vec<String>,
    pub thinking_enabled: bool,
    pub thinking_format: agent_protocol::ThinkingFormat,
    pub input_text: String,
    pub messages: Vec<NormalizedMessage>,
    pub media_artifacts: Vec<MediaArtifact>,
    pub artifacts: Vec<AgentArtifact>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub tool_results: Vec<ToolResultRecord>,
    pub provider_options: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderUsage {
    #[serde(default, alias = "prompt_tokens")]
    pub input_tokens: u32,
    #[serde(default, alias = "completion_tokens")]
    pub output_tokens: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub artifacts: Vec<AgentArtifact>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub usage: ProviderUsage,
}

pub type ProviderStream =
    Pin<Box<dyn Stream<Item = Result<ProviderStreamEvent, ProviderError>> + Send>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderStreamEvent {
    TextDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<ToolCallId>,
        name: Option<String>,
        arguments_delta: String,
    },
    Finish {
        reason: ProviderFinishReason,
    },
    Usage {
        usage: ProviderUsage,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFinishReason {
    Stop,
    ToolCalls,
    FunctionCall,
    Length,
    Other(String),
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider rejected request: {0}")]
    Rejected(String),
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError>;

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream, ProviderError> {
        let response = self.invoke(request).await?;
        Ok(provider_response_stream(response))
    }
}

#[async_trait]
impl<T> ModelProvider for Arc<T>
where
    T: ModelProvider + ?Sized,
{
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        (**self).invoke(request).await
    }

    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream, ProviderError> {
        (**self).stream(request).await
    }
}

pub fn provider_response_stream(response: ProviderResponse) -> ProviderStream {
    let mut events = Vec::new();
    let usage = response.usage.clone();
    if response.tool_calls.is_empty() {
        for artifact in response.artifacts {
            if let AgentArtifact::Text { text, .. } = artifact {
                events.push(Ok(ProviderStreamEvent::TextDelta { text }));
            }
        }
        if usage != ProviderUsage::default() {
            events.push(Ok(ProviderStreamEvent::Usage { usage }));
        }
        events.push(Ok(ProviderStreamEvent::Finish {
            reason: ProviderFinishReason::Stop,
        }));
    } else {
        for (index, call) in response.tool_calls.into_iter().enumerate() {
            events.push(Ok(ProviderStreamEvent::ToolCallDelta {
                index,
                id: Some(call.tool_call_id),
                name: Some(call.tool_name),
                arguments_delta: call.arguments_json.to_string(),
            }));
        }
        if usage != ProviderUsage::default() {
            events.push(Ok(ProviderStreamEvent::Usage { usage }));
        }
        events.push(Ok(ProviderStreamEvent::Finish {
            reason: ProviderFinishReason::ToolCalls,
        }));
    }

    Box::pin(stream::iter(events))
}

pub fn crate_ready() -> bool {
    true
}
