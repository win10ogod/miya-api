# Miya API

Miya API 是一個以 Rust 實作的 OpenAI Chat Completions 與 Anthropic Messages 相容多代理 API 閘道。API 使用者仍然只送出一次標準相容請求；後端會依 `reasoning.effort` 以 deterministic orchestration 將任務拆解為有界子任務，並行派發給子代理，最後由 synthesizer 彙整為單一可用答案。

本專案的目標不是做「多個 chatbot 互聊」的展示型 agent playground，而是提供可商用 API 所需要的核心行為：

- deterministic orchestration
- bounded agent execution
- structured intermediate artifacts
- scoped tool-call accounting
- verification loop
- final synthesis
- request-level isolation
- multimodal image input
- OpenAI/Anthropic tool-call 相容輸出
- true provider token streaming for direct/root streaming paths
- optional encrypted sub-agent state disclosure
- SurrealKV-backed rewindable long-context store and context-pack cache

## Status

目前這是一個可執行的 Rust workspace，包含完整 API server、provider abstraction、OpenAI/Anthropic provider adapters、bounded multi-agent kernel、SurrealKV context store 與測試覆蓋。它已經可以接 OpenAI-compatible 或 Anthropic-compatible 後端，例如本機 `http://localhost:8000/v1`。

仍需注意：

- 伺服器只代理或編排模型呼叫，不會在後端任意執行使用者定義工具。工具呼叫會以 OpenAI/Anthropic 相容格式回給前端，由前端或客戶端執行工具後再把 tool result 送回。
- 高推理難度會增加可用 agent budget 與 deterministic coverage，但最終答案品質仍取決於上游模型是否遵守結構化 spawn/synthesis 指令。
- full multi-agent orchestration 的 streaming 會在完成 orchestration 後輸出相容 SSE；只有 `reasoning.effort=none` 與不需完整 orchestration 的 root path 會直接轉發 provider token streaming。

## Architecture

```text
client
  |
  | OpenAI / Anthropic compatible request
  v
api-server
  |
  | normalize request, image input, tools, thinking mode, reasoning effort
  v
agent-kernel
  |
  | leader -> bounded spawn plan -> parallel child agents -> synthesizer
  v
provider-core
  |
  +-- provider-openai      -> /v1/chat/completions
  +-- provider-anthropic   -> /v1/messages
  +-- MockProvider         -> deterministic local tests
```

Workspace layout:

```text
crates/
  agent-protocol/      Core IDs, scoped artifacts, task graph, tool ledger, normalized requests
  agent-kernel/        Orchestration, spawn validation, artifact store, tool ledger, synthesis
  provider-core/       Provider-neutral invoke/stream trait and normalized stream events
  provider-openai/     OpenAI-compatible upstream adapter and SSE parser
  provider-anthropic/  Anthropic-compatible upstream adapter and SSE parser
  context-store/       SurrealKV-backed rewindable context and context-pack cache
  api-server/          Axum routes, request normalization, response/SSE formatting
```

## Request Lifecycle

1. Client sends one normal OpenAI or Anthropic request.
2. `api-server` parses it into `NormalizedRequest`.
3. Images become scoped `MediaArtifact` records.
4. Tool definitions and tool results become structured protocol records.
5. `reasoning.effort` selects the agent budget.
6. The leader/root agent receives preserved user system instructions plus an orchestration policy.
7. The leader may emit a structured `SpawnPlan`.
8. The kernel validates spawn depth, total agent count, artifact scope and budgets.
9. Child agents run in parallel through `join_all`.
10. Child outputs are stored as internal artifacts; their state is AES-256-GCM encrypted.
11. Root-visible unresolved tool calls are returned to the user if tools are required.
12. If no root tool call remains, a synthesizer returns one natural final answer.
13. Optional context metadata records the exchange into SurrealKV.

Internal child-agent reasoning, tool calls and raw outputs are not exposed by default. Only root-visible tool calls and the final synthesized answer are public unless the caller explicitly asks for encrypted sub-agent state.

## API Compatibility

### OpenAI-compatible endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/chat/completions/batch
```

Supported request features:

- `messages`
- string content and content-part arrays
- `image_url` with remote URL or `data:*;base64,...`
- `tools`
- `tool_choice`
- legacy `functions`
- legacy `function_call`
- assistant `tool_calls` history
- `tool` messages with `tool_call_id`
- `parallel_tool_calls`
- `stream`
- `thinking`, `enable_thinking`, `preserve_thinking`, `chat_template_kwargs`
- `reasoning.effort`
- `metadata`

### Anthropic-compatible endpoints

```text
POST /v1/messages
POST /v1/messages/batch
```

Supported request features:

- `system`
- `messages`
- text content
- content block arrays
- image blocks with base64 source
- image blocks with URL source
- `tools`
- `tool_choice`
- `tool_use` history
- `tool_result` history
- `stream`
- `thinking`
- `reasoning.effort`
- `metadata`

Batch endpoints run items concurrently and preserve per-item response isolation. Batch size is capped at 64 requests.

## Reasoning Effort

`reasoning.effort` controls whether the gateway runs direct provider mode or multi-agent orchestration, and how much parallel agent coverage the kernel allows.

| effort | behavior | max agents per request | target child agents |
| --- | --- | ---: | ---: |
| `none` | direct upstream request, no agent orchestration | 0 | 0 |
| `low` | compact orchestration | 4 | 3 |
| `medium` | compact orchestration, default | 4 | 3 |
| `high` | broader decomposition and verification | 16 | 15 |
| `xhigh` | maximum bounded decomposition | 32 | 31 |

The root agent receives an explicit orchestration policy in its system instructions:

```text
reasoning_effort=<level>
max_agents_per_request=<N>
max_parallel_agents=<N>
target_parallel_agents=<N-1>
max_spawn_depth=<N>
max_total_tool_calls=<N>
max_total_tokens=<N>
```

The test suite includes deterministic coverage evaluation proving that higher effort increases actual task coverage instead of only changing metadata:

- `low` reaches coverage score 3.
- `high` reaches coverage score 8 in the bounded eval.
- `xhigh` is asserted to be at least as strong as `high`.
- API route tests verify higher effort also produces more encrypted child-agent state externally.

## Thinking Modes

The server normalizes several thinking-mode conventions.

OpenAI-compatible request fields:

```json
{
  "enable_thinking": true,
  "preserve_thinking": true,
  "chat_template_kwargs": {
    "enable_thinking": true,
    "preserve_thinking": true
  },
  "thinking": { "type": "enabled" },
  "reasoning": { "enabled": true, "effort": "high" },
  "metadata": {
    "thinking_mode": true,
    "thinking_format": "qwen_chat_template"
  }
}
```

Anthropic-compatible request fields:

```json
{
  "thinking": { "type": "enabled" },
  "reasoning": { "effort": "high" },
  "metadata": {
    "thinking_mode": true,
    "thinking_format": "gemma_system_token"
  }
}
```

Supported `metadata.thinking_format` values:

- `qwen_chat_template`
- `qwen_dashscope`
- `gemma_system_token`

When `thinking_format` is omitted:

- model names containing `qwen` default to Qwen chat-template style.
- model names containing `gemma` default to Gemma system-token style.
- other models use provider auto behavior.

The gateway strips provider reasoning/thinking blocks from public direct responses where possible, so hidden thinking does not leak into user-visible output.

## Tool Calls

Tool calls are accounted by `ToolLedger` using `(IsolationKey, ToolCallId)`. This prevents tool results from one request or conversation from resolving another request's pending tool call.

OpenAI tool response shape:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call-1",
        "type": "function",
        "function": {
          "name": "lookup",
          "arguments": "{\"query\":\"required\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Anthropic tool response shape:

```json
{
  "content": [{
    "type": "tool_use",
    "id": "call-1",
    "name": "lookup",
    "input": { "query": "required" }
  }],
  "stop_reason": "tool_use"
}
```

Only root agent tool calls are public. Child-agent tool calls are internal state and are encrypted if the caller opts into state inclusion.

## Multimodal Image Input

OpenAI image input:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Inspect this image." },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,AAAA"
      }
    }
  ]
}
```

Anthropic image input:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Inspect this image." },
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAAA"
      }
    }
  ]
}
```

The normalizer assigns every image to a scoped `MediaArtifact` with SHA-256 hash, byte length and source type. Child agents can receive images only through scoped artifact references approved by the kernel.

## Long Context And Cache

`context-store` uses SurrealKV with versioning enabled. It supports:

- append-only context chunks
- revision rewind
- query-weighted retrieval
- recent-tail retention
- 1 MiB default assembly window
- max chunk controls
- context-pack cache
- common-prefix reuse plus tail append

Enable persistent context by passing `metadata.context`:

```json
{
  "metadata": {
    "context": {
      "id": "roleplay-session-001",
      "query": "violet launch code",
      "max_context_bytes": 1048576,
      "max_chunks": 128,
      "recent_tail_chunks": 12,
      "cache": true,
      "append": true,
      "include_report": true
    }
  }
}
```

Useful context fields:

| field | meaning |
| --- | --- |
| `id` or `context_id` | logical persistent context key |
| `query` | retrieval query; defaults to latest user text |
| `rewind_revision` or `revision` | read an older context revision |
| `max_context_bytes` | context assembly byte ceiling, clamped to 1 MiB |
| `max_chunks` | max selected chunks, clamped to 128 |
| `recent_tail_chunks` | always keep recent tail in candidate set |
| `cache` | enable context-pack cache |
| `append` or `append_current` | append current request and final answer |
| `include_report` | add `context_cache` report to response |

Set `CONTEXT_STORE=disabled` to disable persistent context. By default the store path is `.multi-agent-context/surrealkv`.

## Encrypted Sub-agent State

Sub-agent state is AES-256-GCM encrypted before it can appear in any response.

Opt in:

```json
{
  "metadata": {
    "include_encrypted_subagent_state": true
  }
}
```

Response field:

```json
{
  "encrypted_agent_state": [{
    "task_id": "child-1",
    "algorithm": "AES-256-GCM",
    "nonce": "...",
    "ciphertext": "...",
    "aad": "request:conversation:task"
  }]
}
```

The encryption key is generated per `KernelRunner` process. Restarting the server rotates it.

## Running Locally

Install Rust stable with edition 2024 support.

Run with deterministic mock provider:

```bash
cargo run -p api-server
```

Default bind address:

```text
127.0.0.1:3000
```

Override it:

```bash
BIND_ADDR=127.0.0.1:8080 cargo run -p api-server
```

Run against an OpenAI-compatible backend:

```bash
MULTI_AGENT_PROVIDER=openai \
OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=local-key \
cargo run -p api-server
```

Run against Anthropic-compatible backend:

```bash
MULTI_AGENT_PROVIDER=anthropic \
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local-key \
ANTHROPIC_VERSION=2023-06-01 \
cargo run -p api-server
```

## Environment Variables

| variable | default | meaning |
| --- | --- | --- |
| `BIND_ADDR` | `127.0.0.1:3000` | API server bind address |
| `MULTI_AGENT_PROVIDER` | `mock` | `mock`, `openai`, or `anthropic` |
| `OPENAI_API_KEY` | required for OpenAI provider | bearer token |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `ANTHROPIC_API_KEY` | required for Anthropic provider | `x-api-key` value |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic-compatible base URL |
| `ANTHROPIC_VERSION` | `2023-06-01` | Anthropic API version header |
| `CONTEXT_STORE` | enabled | set to `disabled` to disable SurrealKV context |
| `CONTEXT_STORE_PATH` | `.multi-agent-context/surrealkv` | SurrealKV storage path |

## Examples

### OpenAI Chat Completions

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "Qwen3.6-27B",
    "reasoning": { "effort": "high" },
    "metadata": {
      "thinking_mode": true,
      "thinking_format": "qwen_chat_template"
    },
    "messages": [
      { "role": "system", "content": "Answer in Traditional Chinese." },
      { "role": "user", "content": "Compare the failure modes of three API gateway designs." }
    ]
  }'
```

### OpenAI Streaming

```bash
curl -N http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "mock",
    "stream": true,
    "reasoning": { "effort": "low" },
    "messages": [
      { "role": "user", "content": "Give a concise answer." }
    ]
  }'
```

### Direct Mode

Use `reasoning.effort=none` to bypass orchestration and forward to the configured provider:

```json
{
  "model": "backend-model",
  "reasoning": { "effort": "none" },
  "messages": [
    { "role": "user", "content": "Generate directly." }
  ]
}
```

### Tool Call Request

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "mock",
    "tools": [{
      "type": "function",
      "function": {
        "name": "lookup",
        "description": "Look up external data.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": { "type": "string" }
          },
          "required": ["query"]
        }
      }
    }],
    "messages": [
      { "role": "user", "content": "Use a tool." }
    ]
  }'
```

Submit the tool result:

```json
{
  "model": "mock",
  "messages": [
    { "role": "user", "content": "Use a tool." },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call-1",
        "type": "function",
        "function": {
          "name": "lookup",
          "arguments": "{\"query\":\"required\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call-1",
      "content": "{\"answer\":\"tool result\"}"
    }
  ]
}
```

### Anthropic Messages

```bash
curl http://127.0.0.1:3000/v1/messages \
  -H "content-type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "system": "Answer in Traditional Chinese.",
    "thinking": { "type": "enabled" },
    "reasoning": { "effort": "high" },
    "metadata": {
      "thinking_format": "gemma_system_token"
    },
    "messages": [
      {
        "role": "user",
        "content": "Design a verification strategy for a multi-agent API."
      }
    ]
  }'
```

### Batch

```bash
curl http://127.0.0.1:3000/v1/chat/completions/batch \
  -H "content-type: application/json" \
  -d '{
    "requests": [
      {
        "model": "mock",
        "messages": [{ "role": "user", "content": "first" }]
      },
      {
        "model": "mock",
        "messages": [{ "role": "user", "content": "second" }]
      }
    ]
  }'
```

## Provider Contract

Provider adapters implement:

```rust
#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn invoke(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError>;
    async fn stream(&self, request: ProviderRequest) -> Result<ProviderStream, ProviderError>;
}
```

`ProviderStreamEvent` normalizes upstream SSE into:

- `TextDelta`
- `ToolCallDelta`
- `Finish`
- `Usage`

This makes OpenAI and Anthropic streaming behavior share one internal event contract while preserving public response shape at the API edge.

## Security And Isolation

Core isolation rules:

- Every request receives a fresh `RequestId`.
- Conversation identity is represented by a `ConversationFingerprint`.
- Artifact and tool lookup requires the full `IsolationKey`.
- Child tasks cannot reference artifacts from other requests.
- Tool results cannot resolve calls from another scope.
- Spawn depth and agent count are bounded.
- Total tool-call budget is enforced.
- Child-agent state is encrypted before optional response inclusion.
- Final synthesis prompt explicitly forbids exposing internal artifacts, sub-agent state and orchestration details.

These rules are enforced by typed protocol structures and kernel validators, not by prompt text alone.

## Verification

Common checks:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Focused checks:

```bash
cargo test -p agent-protocol
cargo test -p agent-kernel
cargo test -p provider-openai
cargo test -p provider-anthropic
cargo test -p context-store
cargo test -p api-server
```

The current test suite covers:

- protocol serialization
- reasoning effort to agent budget mapping
- scoped artifact isolation
- scoped tool ledger behavior
- spawn depth and agent-count rejection
- bounded spawn execution
- encrypted sub-agent state
- OpenAI request normalization
- Anthropic request normalization
- image input normalization
- tool call and tool result compatibility
- legacy OpenAI function compatibility
- OpenAI and Anthropic route responses
- SSE formatting and upstream SSE parsing
- batch concurrency and response isolation
- SurrealKV context rewind
- context-pack cache reuse
- deterministic high-effort coverage improvement

## License

Miya API is released under the MIT License. See [LICENSE](LICENSE).
