# Miya API

Miya API 是一個以 Rust 實作的 OpenAI Responses、OpenAI Chat Completions 與 Anthropic Messages 相容多代理 API 閘道。API 使用者仍然只送出一次標準相容請求；後端會由 leader/planner 模型依 `reasoning.effort` 和實際任務內容產生結構化 `spawn_plan`，內核再以有界並行派發子代理，最後由 synthesizer 彙整為單一可用答案。對前端來說它仍然像單一 OpenAI/Anthropic 相容 API；對後端來說它是可控的多請求 fan-out、上下文快取、工具帳本與最終合成管線。

本專案的目標不是做「多個 chatbot 互聊」的展示型 agent playground，而是提供可商用 API 所需要的核心行為：

- model-selected orchestration with deterministic kernel execution
- bounded agent execution
- structured intermediate artifacts
- scoped tool-call accounting
- independent semantic verification with a bounded repair loop
- final synthesis
- request-level isolation
- multimodal image, audio, file, document, citation, and provider-native content blocks
- OpenAI/Anthropic tool-call 相容輸出
- true provider token streaming for direct/root streaming paths
- structured usage telemetry with output-token accounting
- bounded concurrent provider calls, default medium tier 16 child agents with 4 in-flight workers
- optional encrypted sub-agent state disclosure
- SurrealKV-backed rewindable long-context store and context-pack cache
- header/metadata-based multi-tenant isolation

## Status

目前這是一個可執行的 Rust workspace，包含完整 API server、provider abstraction、OpenAI/Anthropic provider adapters、bounded multi-agent kernel、SurrealKV context store 與測試覆蓋。它已經可以接 OpenAI-compatible 或 Anthropic-compatible 後端，例如本機 `http://localhost:8000/v1`、LM Studio/Ollama/vLLM 類 OpenAI-compatible endpoint，或真正的 Anthropic-compatible endpoint。

商用部署時的核心邊界是：

- Client compatibility: 前端仍使用標準 `/v1/responses`、`/v1/chat/completions`、`/v1/messages`、stream、tool-call 格式。
- Provider efficiency: 子代理會以有界並行向同一後端發送多個標準 provider request，預設最多 4 個 child-agent request 同時在飛。
- Isolation: tenant、request、conversation、artifact、tool-call、context cache key 都有明確隔離，避免批量請求或多用戶場景串話。
- Cache discipline: context-pack cache 依 tenant/context/revision/retrieval policy/model/tools/thinking/provider options/system hash 切分，避免不同後端格式、模型或工具 schema 共用錯誤上下文。
- Observability: stdout JSONL telemetry、Prometheus `/metrics`、W3C trace-context 與可選 OTLP/OpenTelemetry tracing 記錄 provider retry/circuit、durable job、token、agent 與 context 行為；可選 training trace 記錄輸入輸出與中間 artifact。

仍需注意：

- 伺服器只代理或編排模型呼叫，不會在後端任意執行使用者定義工具。工具呼叫會以 OpenAI/Anthropic 相容格式回給前端，由前端或客戶端執行工具後再把 tool result 送回。
- 高推理難度會增加可用 agent coverage 與模型規劃上限；token 用量會被記錄為 telemetry，不會作為阻止子代理生成的條件。
- `low`、`medium`、`high`、`xhigh` 的 streaming 都會保留完整 4/16/32/64-agent orchestration，完成後輸出相容 SSE；只有明確的 `reasoning.effort=none` 走 direct provider streaming。等待完整 orchestration 時會持續送出 SSE comment heartbeat。

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
  |                                      \-> reasoning-summary agent (public thought summary only)
  v
provider-core
  |
  +-- provider-openai      -> /v1/chat/completions
  +-- provider-anthropic   -> /v1/messages
  +-- MockProvider         -> reproducible local tests
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
5. `reasoning.effort` selects the agent coverage tier.
6. The leader/root agent receives preserved user system instructions plus an orchestration policy.
7. The leader/planner emits a structured `SpawnPlan` that decides the child-agent division for this request.
8. The kernel validates spawn depth, total agent count, artifact scope and tool-call budgets.
9. Child agents run through a bounded concurrent fan-out queue; results are sorted back into stable task order before verification.
10. Child outputs are stored as internal artifacts; their state is AES-256-GCM encrypted.
11. Root-visible unresolved tool calls are returned to the user if tools are required.
12. If public reasoning is enabled, one dedicated `reasoning-summary` agent summarizes worker outputs for the frontend reasoning field only.
13. If no root tool call remains, a synthesizer reads the original worker artifacts and returns one natural final answer.
14. An independent semantic verifier checks instruction coverage, contradictions, unsupported claims, and artifact coverage; failed verdicts enter a configured bounded repair-and-recheck loop.
15. Files, batches, Message Batches, and background Responses use a durable filesystem object/blob store and process-wide asynchronous job scheduler with cancellation and restart recovery.
16. Optional context metadata records the exchange into SurrealKV.
17. The API emits structured telemetry, Prometheus counters, and OpenTelemetry spans.

Internal child-agent reasoning, tool calls and raw outputs are not exposed by default. Only root-visible tool calls and the final synthesized answer are public unless the caller explicitly asks for encrypted sub-agent state.

## API Compatibility

### OpenAI-compatible endpoints

```text
GET  /health
GET  /models
GET  /v1/models
GET  /v1/v1/models
GET  /v1/models/{model_id}
GET  /metrics
POST /v1/files
GET  /v1/files
GET  /v1/files/{file_id}
GET  /v1/files/{file_id}/content
DELETE /v1/files/{file_id}
POST /v1/batches
GET  /v1/batches
GET  /v1/batches/{batch_id}
POST /v1/batches/{batch_id}/cancel
POST /completions
POST /v1/completions
POST /v1/v1/completions
POST /chat/completions
POST /v1/chat/completions
POST /v1/v1/chat/completions
POST /responses
GET  /responses
POST /v1/responses
GET  /v1/responses
POST /v1/v1/responses
GET  /v1/v1/responses
GET  /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
GET  /v1/responses/{response_id}/input_items
POST /v1/responses/input_tokens
POST /v1/responses/compact
```

Canonical OpenAI-compatible base URL is `/v1`. The root and double-`/v1` aliases exist for frontends that either omit `/v1` or append `/v1` themselves. 非標準的 `/chat/completions/batch` aliases 不再提供。官方 Files/Batch 流程已實作：先以 multipart 上傳 `purpose=batch` JSONL 至 `/v1/files`，再建立 `/v1/batches`，並透過 retrieve/list/cancel 與 output/error file lifecycle 非同步處理。此閘道目前接受其實際可執行的 batch endpoints：`/v1/chat/completions`、`/v1/completions`、`/v1/responses`；其他官方 endpoint 會明確回傳 `400`，不會假裝已處理。Metadata 與 blobs 以 atomic file replacement 寫入 `MIYA_DATA_DIR`；restart recovery 採 at-least-once 語意，因此 crash 時仍在飛的單項模型請求可能被重新執行，但已完成的 output/error file 不會被回報成部分完成檔案。

In practice:

- if a frontend asks for "OpenAI API URL" or "endpoint", use `http://localhost:3100/v1`;
- if a frontend asks for "base URL" and appends `/v1` internally, use `http://localhost:3100`.

### Shared API Key

Miya 使用單一 deployment-wide shared key，不要求每位使用者各自提供 key。設定 `MIYA_API_KEY` 後：

- OpenAI SDK 使用相同 key 作為 `Authorization: Bearer ...`；
- Anthropic SDK 使用相同 key 作為 `x-api-key: ...`；
- `/health` 保持公開，其他 API 路徑要求 shared key；
- `tenant_id` 仍用於工作區、context 與 telemetry 隔離，不會被解讀成獨立 credential。

```python
from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(base_url="http://127.0.0.1:3100/v1", api_key="miya-local-key")
anthropic_client = Anthropic(base_url="http://127.0.0.1:3100", api_key="miya-local-key")
```

若未設定 `MIYA_API_KEY`，為維持本機開發相容性，shared-key 驗證不會啟用。Windows launcher 預設提供共同 key `miya-local-key`，部署到其他主機時應改成自己的 deployment key。

Supported request features:

- `messages`
- string content and content-part arrays
- `image_url` with remote URL or `data:*;base64,...`
- OpenAI `input_audio` and `file` content parts, including Responses `input_file` conversion
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
- provider model options such as `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `response_format`, `seed`, `stop`, `logit_bias`, `logprobs`, `top_logprobs`, `n`, `modalities`, `audio`, `prediction`, `service_tier`, `stream_options`, `verbosity`, `user`, `safety_identifier`, and backend-specific extra fields

### OpenAI Responses-compatible API

`POST /v1/responses` is implemented as a compatibility layer on top of the same Rust orchestration kernel and provider adapters. It accepts the Responses request shape, normalizes it into the internal chat representation, runs either direct provider passthrough or model-selected multi-agent orchestration, then emits Responses-shaped output items and SSE events.

Supported Responses features:

- string `input`
- message input items
- input content parts: `input_text`, `text`, `input_image`, `image_url`, `input_audio`, `input_file`, `file`
- output/tool history items: `function_call`, `function_call_output`, `custom_tool_call`, `custom_tool_call_output`, `local_shell_call`, `shell_call`, `apply_patch_call`, and matching output items
- `instructions`
- `tools`, including Responses-style top-level function tools and OpenAI Chat-style `function` tools
- `tool_choice`
- `parallel_tool_calls`
- `stream`
- durable `background=true` execution, polling, and actual in-flight cancellation
- `store`
- `previous_response_id`
- `metadata` tenant/request/conversation isolation
- `reasoning_effort`, `reasoning.effort`, and effort-suffixed model aliases
- `max_output_tokens`, `temperature`, `top_p`, `top_logprobs`, `service_tier`, `safety_identifier`, prompt-cache options, `text.response_format`, `text.verbosity`, and backend-specific extra model options

Stored Responses are tenant-scoped and backed by SurrealKV when `CONTEXT_STORE_PATH` is configured, with an in-memory mirror for fast local reads. `previous_response_id` reloads the stored conversation messages for the same tenant only, so different tenants cannot continue or retrieve each other's response chains.

Streaming emits Responses-style SSE events:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.function_call_arguments.delta
response.function_call_arguments.done
response.output_item.done
response.completed
data: [DONE]
```

`GET /v1/responses/{response_id}` retrieves a stored response, `GET /v1/responses/{response_id}/input_items` returns the original input items, and `DELETE /v1/responses/{response_id}` removes tenant-scoped stored state. `background=true` 會立即建立 `queued` response，durable worker 隨後更新為 `in_progress`／`completed`／`failed`；`POST /v1/responses/{response_id}/cancel` 會持久化取消意圖並中止目前 execution future。`POST /v1/responses/input_tokens` provides a local token estimate for frontend budgeting. `POST /v1/responses/compact` returns a compacted artifact envelope suitable for clients expecting the endpoint; semantic model-driven compaction is intentionally kept separate from this compatibility response. `conversation`、prompt templates 與自動 truncation 仍會明確回傳 `400`，不會被靜默忽略。

Legacy OpenAI Completions compatibility:

- `POST /v1/completions` accepts `prompt` requests from text-completion frontends such as SillyTavern `api_type: generic`.
- The gateway converts `prompt` into an internal user message, so it can still use the same provider adapters, context store, reasoning effort, telemetry and cache isolation.
- Responses use legacy `text_completion` shape with `choices[].text`; streaming emits text-completion SSE chunks and ends with `data: [DONE]`.
- Completion model parameters such as `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `logprobs`, `n`, and `best_of` are preserved as provider options.

### Anthropic-compatible endpoints

```text
POST /messages
POST /v1/messages
POST /v1/v1/messages
POST /messages/count_tokens
POST /v1/messages/count_tokens
POST /v1/v1/messages/count_tokens
POST /v1/messages/batches
GET  /v1/messages/batches
GET  /v1/messages/batches/{message_batch_id}
POST /v1/messages/batches/{message_batch_id}/cancel
DELETE /v1/messages/batches/{message_batch_id}
GET  /v1/messages/batches/{message_batch_id}/results
```

Supported request features:

- `system`
- `messages`
- text content
- content block arrays
- image blocks with base64 source
- image blocks with URL source
- document and search-result blocks
- citation-bearing text blocks
- server-tool use/result, code execution, web fetch/search, tool search, container upload, thinking, and other current provider-native content blocks
- `tools`
- `tool_choice`
- `tool_use` history
- `tool_result` history
- `stream`
- `thinking`
- `reasoning_effort`, `reasoning.effort`
- `metadata`
- provider model options such as `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `metadata`, service-tier/provider beta fields, and backend-specific extra fields

`POST /v1/messages/count_tokens` 提供符合目前 Anthropic SDK `MessageTokensCount` shape 的本機估算，包含 system、messages、tools 與 image block 基準成本。它不是 provider tokenizer 的精確 billing 值。

舊有同步 `/messages/batch` aliases 已移除。`/v1/messages/batches` 現在使用官方 `{custom_id, params}` contract，建立後立即非同步執行；retrieve/list/cancel/delete 及 `results_url` JSONL 下載均可由目前 Anthropic SDK 使用。Batch metadata、輸入 payload、結果與取消狀態會寫入 durable store，server restart 後會重新排程未完成工作。

## Provider Parameter Pass-Through

Miya API only consumes gateway/orchestration parameters. Model configuration parameters are preserved and forwarded to the configured backend provider so existing OpenAI Chat Completions and Anthropic Messages frontends keep working. 本次相容性基準參考 `openai-python 2.52.0` 與 `anthropic 0.120.2`。Public synthesizer 與 root-visible tool continuation 會保留呼叫者的 output cap 與 stop conditions；只有 planner、worker 與 reasoning-summary 等內部呼叫使用獨立的 internal generation floor。

Gateway-consumed parameters:

| location | consumed by gateway |
| --- | --- |
| `reasoning_effort`, `reasoning.effort` | selects direct mode or multi-agent coverage tier: `none`, `low`, `medium`, `high`, `xhigh` |
| `metadata.tenant_id`, tenant aliases | tenant isolation and per-tenant concurrency |
| `metadata.request_id`, request aliases | request correlation and telemetry |
| `metadata.conversation_id`, thread/session aliases | conversation fingerprint and scoped artifacts |
| `metadata.context` | SurrealKV long-context assembly and append policy |
| `metadata.agent`, `metadata.orchestration`, `metadata.max_parallel_agents`, `metadata.parallel_agents` | per-request child-agent concurrency |
| `metadata.thinking_mode`, `metadata.thinking_format` | Qwen/Gemma thinking format adaptation |
| `metadata.include_context_report`, `metadata.include_encrypted_subagent_state` | optional response-side diagnostics |

### Effort Model Aliases

Every non-mock model exposed by `MULTI_AGENT_MODELS` also gets effort-suffixed model aliases:

| alias suffix | gateway effort | child-agent target |
| --- | --- | --- |
| `-none` | `none` | 0, direct provider passthrough |
| `-low` | `low` | 4 |
| `-medium` | `medium` | 16 |
| `-high` | `high` | 32 |
| `-xhigh` | `xhigh` | 64 |

For example, if `MULTI_AGENT_MODELS=local-model,mock`, `/v1/models` exposes `local-model`, `local-model-none`, `local-model-low`, `local-model-medium`, `local-model-high`, `local-model-xhigh`, and `mock`.

The alias is gateway-only. A request using `model: "local-model-high"` is sent to the backend as `model: "local-model"` while Miya uses `reasoning.effort=high`. The alias wins over conflicting request fields such as top-level `reasoning_effort` and nested `reasoning.effort`, so frontends that cannot set custom reasoning fields can select the tier purely through the model name. Without a model suffix, top-level `reasoning_effort` is preferred over nested `reasoning.effort`.

Everything else is treated as provider configuration. For OpenAI-compatible requests, unknown top-level JSON fields are captured and merged into the upstream `/chat/completions` body after normalization. For Anthropic-compatible requests, unknown top-level JSON fields are captured and merged into the upstream `/messages` body. This includes both official model controls and OpenAI-compatible local-server extensions.

Reserved transport/core fields cannot be overridden by pass-through because the gateway must own normalization and public response shape:

| provider | reserved fields |
| --- | --- |
| OpenAI-compatible | `model`, `messages`, `tools`, `tool_choice`, `parallel_tool_calls`, `stream`, `functions`, `function_call` |
| Anthropic-compatible | `model`, `system`, `messages`, `tools`, `tool_choice`, `stream` |

`metadata` is handled specially: gateway-only keys are stripped before provider forwarding, but unrelated metadata keys are preserved. For example `metadata.foo` is forwarded, while `metadata.tenant_id` and `metadata.context` are not. `reasoning` is also handled specially: nested `reasoning.effort` is removed before forwarding because it controls Miya's agent coverage tier, while other nested reasoning provider options such as `reasoning.summary` are preserved. Top-level `reasoning_effort` is also consumed by Miya and stripped before provider forwarding.

Example:

```json
{
  "model": "local-model",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_completion_tokens": 1024,
  "response_format": { "type": "json_object" },
  "stream_options": { "include_usage": true },
  "reasoning_effort": "low",
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  },
  "metadata": {
    "tenant_id": "tenant-a",
    "context": { "id": "long-context-session-001", "cache": true },
    "foo": "bar"
  },
  "messages": [
    { "role": "user", "content": "Return JSON." }
  ]
}
```

The gateway uses top-level `reasoning_effort=low`, tenant/context metadata and cache policy. The provider still receives `temperature`, `top_p`, `max_completion_tokens`, `response_format`, `stream_options`, `reasoning.summary`, and `metadata.foo`.

## Multi-User Isolation

The gateway scopes requests by tenant, request and conversation identity before entering the agent kernel:

- `tenant_id` isolates artifacts, tool-call ledger records, encrypted sub-agent state and SurrealKV context/cache keys.
- `request_id` keeps concurrent requests distinct even when they share a conversation.
- `conversation_id` produces a stable conversation fingerprint for encryption AAD and scoped artifacts.

Identity can be supplied through HTTP headers:

| header | purpose |
| --- | --- |
| `x-tenant-id` | tenant/workspace isolation key |
| `x-organization-id` | fallback tenant key |
| `x-project-id` | fallback tenant key |
| `x-user-id` | fallback tenant key |
| `x-request-id` | caller trace/request id |
| `x-correlation-id` | fallback request id |
| `x-conversation-id` | stable conversation/thread id |
| `x-thread-id` | fallback conversation id |

The same identity can also be supplied per request through `metadata`, which is especially useful for batch calls:

```json
{
  "model": "local-model",
  "metadata": {
    "tenant_id": "tenant-a",
    "request_id": "req-2026-05-13-0001",
    "conversation_id": "thread-7",
    "context": {
      "id": "shared-long-context",
      "include_report": true
    }
  },
  "messages": [{ "role": "user", "content": "Continue." }]
}
```

`metadata` overrides headers, headers override the default tenant `default`. Unsafe or very long identity values are internally hashed into bounded ASCII components, so user-facing IDs can be accepted without becoming storage-key material. 自訂 Anthropic batch 擴充會逐項獨立正規化，因此同一個 batch 可安全包含不同 tenant 的 requests。

Production routers also apply a per-tenant concurrency limiter. `TENANT_MAX_CONCURRENT_REQUESTS` defaults to the medium tier, `16`, when the router is built from environment variables. Set it to `0` to disable the limiter. Requests beyond the same tenant's limit wait on that tenant's semaphore; other tenants continue running independently. This keeps one tenant's large Anthropic batch extension call or high-effort workload from monopolizing provider capacity.

## Reasoning Effort

`reasoning.effort` controls whether the gateway runs direct provider mode or multi-agent orchestration, and how much child-agent coverage the kernel allows. Agent coverage and provider-call concurrency are intentionally separate: `medium` gives the planner a 16-agent commercial coverage target by default, while the default runtime only runs 4 child-agent backend calls at once so latency improves without unbounded provider pressure. Token usage is accounted and logged, not used as a stop condition for child-agent creation.

| effort | behavior | max agents per request | target child agents | default concurrent child calls |
| --- | --- | ---: | ---: | ---: |
| `none` | direct upstream request, no agent orchestration | 0 | 0 | 0 |
| `low` | compact orchestration | 4 | 4 | 4 |
| `medium` | broader orchestration, default | 16 | 16 | 4 |
| `high` | deep decomposition and verification | 32 | 32 | 4 |
| `xhigh` | maximum bounded decomposition | 64 | 64 | 4 |

Concurrency can be configured globally through environment variables:

```bash
MULTI_AGENT_MAX_PARALLEL_AGENTS=4 \
MIYA_TENANT_QUEUE_TIMEOUT_MS=30000 \
MIYA_PROVIDER_MAX_CONCURRENT=64 \
MIYA_PROVIDER_QUEUE_TIMEOUT_MS=30000 \
MIYA_MAX_CONCURRENT_ORCHESTRATIONS=16 \
MIYA_ORCHESTRATION_QUEUE_TIMEOUT_MS=30000 \
MIYA_REQUEST_TIMEOUT_MS=3600000 \
MIYA_AGENT_TIMEOUT_MS=330000 \
MIYA_STREAM_HEARTBEAT_SECS=10 \
cargo run -p api-server
```

`MULTI_AGENT_MAX_PARALLEL_AGENTS` 限制單一 orchestration 的 child fan-out；`MIYA_PROVIDER_MAX_CONCURRENT` 是跨 request、跨 tenant、同時涵蓋 kernel 與 direct passthrough 的 process-wide provider admission limit。Miya 另在任何 planner/worker 開始前取得完整 orchestration admission；每個請求會依其有效 child parallelism 取得一個或多個加權 slots，避免大量 high/xhigh 請求各自占住部分 provider capacity 後一起 queue timeout。未明確設定 `MIYA_MAX_CONCURRENT_ORCHESTRATIONS` 時，其基準 slots 上限自動推導為 `floor(provider max concurrent / default max parallel agents)`，至少為 1；這只約束同時執行的完整請求，不會減少單一請求的 agent coverage。

Tenant、orchestration 與 provider 排隊均有明確上限。Orchestration 過載回傳帶 `Retry-After` 的 `503 orchestration_overloaded`，provider admission 過載回傳 `503 provider_overloaded`，agent/request 執行超時回傳 `504`。預設 timeout 已為慢速推理模型保留 330 秒單次 agent 與 60 分鐘完整 orchestration；用戶明確設定的 timeout 仍會原樣執行。啟動時會輸出 `runtime_stability_profile`，並在 agent timeout 比 provider timeout 短、或總 request deadline 無法容納 planner/worker/synthesizer 的完整延遲窗口時發出 warning。完整 orchestration 的 SSE 每 10 秒送出標準 comment heartbeat，避免 proxy/load balancer 在模型思考期間因 idle timeout 中斷；heartbeat 不會偽造模型 token 或改變輸出。

The same limit can be overridden per request:

```json
{
  "metadata": {
    "agent": {
      "max_parallel_agents": 2
    }
  }
}
```

Valid per-request aliases are `metadata.max_parallel_agents`, `metadata.parallel_agents`, `metadata.agent.max_parallel_agents`, `metadata.agent.parallel_agents`, `metadata.agent.parallelism`, `metadata.orchestration.max_parallel_agents`, and `metadata.orchestration.parallel_agents`. The runtime clamps the value to `1..max_agents_per_request`; use `reasoning.effort=none` for 0-agent direct mode.

The root agent receives an explicit orchestration policy in its system instructions:

```text
reasoning_effort=<level>
max_agents_per_request=<N>
max_parallel_agents=<configured concurrency>
target_parallel_agents=<N>
max_spawn_depth=<N>
max_total_tool_calls=<N>
token_accounting_reference=<N>
```

The leader may spawn more child tasks than the concurrency limit. The kernel executes those children with bounded parallelism, sends multiple OpenAI/Anthropic-compatible provider calls in parallel, then restores stable task order before writing artifacts, accounting token/tool usage, sealing sub-agent state and invoking final synthesis. If a model accidentally emits 33/34 planner children for the `high=32` contract, the kernel bounds the model plan to exactly 32 instead of failing the whole request. If an `xhigh=64` planner remains at 63 after the bounded model repair loop, the kernel adds the missing independent cross-check from an existing model-selected objective, executes all 64 agents, and records the non-fatal `orchestration_plan_reconciled` verification issue. A text-only worker that returns `spawn_plan` JSON cannot mutate the task graph; its output is preserved as text.

A reasoning model can occasionally consume a completion budget without emitting final text. Final synthesis therefore receives up to two bounded retries only when the prior synthesis is empty. Retries preserve the exact model, thinking mode, provider options, output limit, evidence, and user constraints; successful recovery is reported as `empty_synthesis_recovered`. Three consecutive empty synthesis attempts return a structured upstream `502` instead of a misleading successful response with empty content. No reasoning tier, agent count, token setting, tool, or streaming capability is silently reduced.

The test suite includes model-planned coverage evaluation proving that higher effort increases actual task coverage instead of only changing metadata:

- `low` reaches coverage score 4.
- `high` reaches coverage score 32 in the bounded eval.
- `xhigh` reaches coverage score 64.
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
- model IDs listed in `MIYA_GEMMA_MODELS`, `MULTI_AGENT_GEMMA_MODELS`, or `GEMMA_MODELS` default to Gemma system-token style. This is how local aliases such as `local-gemma-model` are mapped without hardcoding a model ID in the API kernel.
- other models use provider auto behavior.

Public reasoning output is deployment controlled. The exposed reasoning is produced by one dedicated `reasoning-summary` agent from bounded worker artifacts; it is not raw provider hidden reasoning and it does not include child-agent private traces or child-agent tool-call payloads unless a separate encrypted-state diagnostic is explicitly requested. This summary agent is not part of final answer synthesis: the final synthesizer still receives the original worker artifacts, so a short public thought summary cannot compress or shorten the user-facing answer.

| `MIYA_PUBLIC_REASONING` value | behavior |
| --- | --- |
| `always` | default; include the synthesized multi-agent reasoning summary for orchestrated responses even when the frontend did not ask |
| `request` | include public reasoning only when the request asks for it through `include_reasoning`, `include_thinking`, `show_reasoning`, `return_reasoning`, `reasoning.summary`, or an enabled `thinking` block |
| `never` or `strip` | suppress public reasoning even when the frontend asks; return only the final answer/tool-call surface |

Aliases are also accepted through `MIYA_PUBLIC_REASONING_MODE`, `MULTI_AGENT_PUBLIC_REASONING`, or `PUBLIC_REASONING_MODE`. Truthy values such as `on` and `enabled` map to `always`; falsy values such as `off` and `disabled` map to `never`. If none of these environment variables are set, the gateway uses `always`.

For OpenAI Chat Completions, public reasoning is returned as `message.reasoning_content` plus `message.reasoning.summary`, and streaming sends a reasoning delta before the final content delta. For Anthropic Messages, it is returned as a `thinking` content block or `thinking_delta` stream event. For legacy OpenAI Completions, Qwen-style models receive:

```text
<think>
Multi-agent process summary...
</think>

Final answer...
```

Gemma-style models receive:

```text
<|channel>thought
Multi-agent process summary...
<channel|>
Final answer...
```

The gateway strips provider reasoning/thinking blocks from public direct responses where possible, so hidden thinking does not leak into user-visible output. Direct mode with `reasoning.effort=none` has no multi-agent process to summarize.

Final synthesis and provider worker prompts also instruct models to preserve structured output formatting. When an answer contains XML/HTML-like tags, Markdown, fenced code, lists, tables or delimiter-separated blocks, the gateway asks the backend not to minify, collapse line breaks, remove heading spaces, or merge tag-delimited sections. After synthesis, the kernel applies a deterministic generic layout normalization pass that restores obvious line boundaries around XML-like tags, Markdown headings and repeated ASCII field labels without matching domain-specific tag names, characters, or sample text.

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
- cache namespaces that isolate model/tool/thinking/system profiles

Enable persistent context by passing `metadata.context`:

```json
{
  "metadata": {
    "context": {
      "id": "long-context-session-001",
      "query": "violet launch code",
      "max_context_bytes": 1048576,
      "max_chunks": 128,
      "recent_tail_chunks": 12,
      "cache": true,
      "cache_namespace": "project-local-model-tools-v1",
      "append": true,
      "include_report": true
    }
  }
}
```

If `cache_namespace` is omitted, the server generates one from:

- source format: OpenAI Chat Completions or Anthropic Messages
- model name
- thinking enabled/disabled flag
- thinking format, such as `qwen_chat_template`, `qwen_dashscope`, or `gemma_system_token`
- tool definitions, tool choice and `parallel_tool_calls` hash
- provider model options hash, such as sampling, token-limit, response-format, stream-options or backend-specific fields
- system prompt hash

The generated namespace is part of the context-pack policy hash together with retrieval query, byte limit, chunk limit and recent-tail policy. This means the same tenant/context/revision can safely reuse cached common prefixes only when the relevant backend profile is the same. Different models, tools, system prompts, thinking formats or provider model options build independent context packs, avoiding cache pollution in multi-user and multi-backend deployments.

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
| `cache_namespace` or `namespace` | manually isolate cache packs for a backend/profile; auto-generated when omitted |
| `append` or `append_current` | append current request and final answer |
| `include_report` | add `context_cache` report to response |

Set `CONTEXT_STORE=disabled` to disable persistent context. By default the store path is `.multi-agent-context/surrealkv`.

When `include_report` is true, responses include context accounting such as:

```json
{
  "context_cache": {
    "enabled": true,
    "context_id": "long-context-session-001",
    "cache_namespace": "openai_chat|model:local-model|thinking:on:gemma_system_token|tools:...|system:...",
    "revision": 42,
    "included_chunks": 12,
    "included_bytes": 98304,
    "cache_hit": true,
    "base_cache_revision": 40,
    "tail_chunks": 2,
    "stored_revision": 44
  }
}
```

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

Run with reproducible mock provider:

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
MIYA_API_KEY=miya-local-key \
cargo run -p api-server
```

Windows deployment for a local Gemma-format fine-tune:

```powershell
.\scripts\windows\start-miya-api.ps1 `
  -BindAddr "127.0.0.1:3100" `
  -OpenAIBaseUrl "http://YOUR_BACKEND_HOST:PORT/v1" `
  -OpenAIApiKey "local-key" `
  -MiyaApiKey "miya-local-key" `
  -DefaultModel "local-gemma-model" `
  -GemmaModels "local-gemma-model" `
  -TenantMaxConcurrentRequests 16 `
  -MaxParallelAgents 4 `
  -PublicReasoning always `
  -TrainingTrace
```

Windows deployment for a local OpenAI-compatible Qwen backend:

```powershell
.\scripts\windows\start-miya-api.ps1 `
  -BindAddr "127.0.0.1:3100" `
  -OpenAIBaseUrl "http://localhost:8000/v1" `
  -OpenAIApiKey "local-key" `
  -MiyaApiKey "miya-local-key" `
  -DefaultModel "local-qwen-model" `
  -TenantMaxConcurrentRequests 16 `
  -MaxParallelAgents 4 `
  -PublicReasoning always `
  -TrainingTrace
```

Use the host name and port that actually serve `/v1/chat/completions` on your machine.

Smoke test:

```powershell
.\scripts\windows\smoke-miya-api.ps1 `
  -BaseUrl "http://127.0.0.1:3100" `
  -Model "local-qwen-model" `
  -MiyaApiKey "miya-local-key"
```

On this Windows deployment the launcher defaults to `127.0.0.1:3100` because port `3000` is commonly occupied by Docker/WSL relay processes. Use `http://localhost:3100/v1` or `http://127.0.0.1:3100/v1` in OpenAI Chat Completions clients.

For SillyTavern Text Completion with `api_type: generic`, use:

```text
http://127.0.0.1:3100
```

SillyTavern automatically appends `/v1/completions` for that mode. For SillyTavern Chat Completion with `Custom (OpenAI-compatible)`, use:

```text
http://127.0.0.1:3100/v1
```

Invoke one CLI request and print its matching backend telemetry:

```powershell
.\scripts\windows\invoke-miya-api.ps1 `
  -BaseUrl "http://localhost:3100" `
  -Model "local-qwen-model" `
  -MiyaApiKey "miya-local-key" `
  -Effort low `
  -MaxParallelAgents 4 `
  -Message "OK"
```

Watch backend usage records live:

```powershell
.\scripts\windows\watch-miya-api-telemetry.ps1 -Follow
```

Watch training samples live:

```powershell
.\scripts\windows\watch-miya-training-traces.ps1 -Follow
```

Export recorded JSONL samples as a single JSON array for training:

```powershell
.\scripts\windows\export-miya-training-dataset.ps1 `
  -OutputPath "logs\training-dataset.json"
```

Stop the deployed process:

```powershell
.\scripts\windows\stop-miya-api.ps1
```

The Windows launcher sets:

```text
MULTI_AGENT_PROVIDER=openai
OPENAI_BASE_URL=http://YOUR_BACKEND_HOST:PORT/v1
OPENAI_API_KEY=local-key
MIYA_API_KEY=miya-local-key
TENANT_MAX_CONCURRENT_REQUESTS=16
MIYA_TENANT_QUEUE_TIMEOUT_MS=30000
MIYA_PROVIDER_MAX_CONCURRENT=64
MIYA_PROVIDER_QUEUE_TIMEOUT_MS=30000
MIYA_PROVIDER_TIMEOUT_SECS=300
MIYA_PROVIDER_CONNECT_TIMEOUT_SECS=30
MIYA_ORCHESTRATION_QUEUE_TIMEOUT_MS=30000
MIYA_PROVIDER_MAX_RETRIES=2
MIYA_PROVIDER_RETRY_BASE_MS=250
MIYA_PROVIDER_CIRCUIT_FAILURE_THRESHOLD=5
MIYA_PROVIDER_CIRCUIT_COOLDOWN_MS=30000
MIYA_REQUEST_TIMEOUT_MS=3600000
MIYA_AGENT_TIMEOUT_MS=330000
MIYA_STREAM_HEARTBEAT_SECS=10
MIYA_DATA_DIR=.multi-agent-data
MIYA_MAX_CONCURRENT_JOBS=4
MIYA_BATCH_ITEM_CONCURRENCY=8
MIYA_SEMANTIC_VERIFIER=true
MIYA_SEMANTIC_MAX_REPAIR_ATTEMPTS=2
MULTI_AGENT_MAX_PARALLEL_AGENTS=4
MULTI_AGENT_MODELS=local-model,mock
MIYA_GEMMA_MODELS=local-gemma-model
MIYA_PUBLIC_REASONING=always
TRAINING_TRACE=enabled
TRAINING_TRACE_PATH=logs\training-traces.jsonl
```

For a Windows Gemma-format deployment, `MIYA_GEMMA_MODELS=local-gemma-model` tells the gateway to keep thinking enabled and use `GemmaSystemToken` formatting. The API kernel does not hardcode that model ID; change the environment value when deploying another Gemma-format alias.

The `/v1/models` response expands that base list with effort aliases. With the Windows defaults above, frontends can select `local-model-none`, `local-model-low`, `local-model-medium`, `local-model-high`, or `local-model-xhigh`; all are forwarded upstream as `local-model`.

Windows stdout telemetry is written to:

```text
logs\api-server.out.log
```

Run against Anthropic-compatible backend:

```bash
MULTI_AGENT_PROVIDER=anthropic \
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local-key \
ANTHROPIC_VERSION=2023-06-01 \
MIYA_API_KEY=miya-local-key \
cargo run -p api-server
```

## Environment Variables

| variable | default | meaning |
| --- | --- | --- |
| `BIND_ADDR` | `127.0.0.1:3000` | API server bind address |
| `MULTI_AGENT_PROVIDER` | `mock` | `mock`, `openai`, or `anthropic` |
| `MIYA_API_KEY` | disabled when unset | one deployment-wide shared client key; accepted as OpenAI Bearer or Anthropic `x-api-key`; alias: `MIYA_SHARED_API_KEY` |
| `OPENAI_API_KEY` | required for OpenAI provider | upstream provider bearer token; independent from `MIYA_API_KEY` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `ANTHROPIC_API_KEY` | required for Anthropic provider | `x-api-key` value |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic-compatible base URL |
| `ANTHROPIC_VERSION` | `2023-06-01` | Anthropic API version header |
| `CONTEXT_STORE` | enabled | set to `disabled` to disable SurrealKV context |
| `CONTEXT_STORE_PATH` | `.multi-agent-context/surrealkv` | SurrealKV storage path |
| `TENANT_MAX_CONCURRENT_REQUESTS` | `16` in env-built router | per-tenant concurrency cap; `0` disables |
| `MIYA_TENANT_QUEUE_TIMEOUT_MS` | `30000` | bounded wait for a per-tenant request slot; overload returns `503` plus `Retry-After` instead of hanging indefinitely |
| `MULTI_AGENT_MAX_PARALLEL_AGENTS` | `4` | max concurrent child-agent provider calls; aliases: `MIYA_MAX_PARALLEL_AGENTS`, `MAX_PARALLEL_AGENTS` |
| `MIYA_PROVIDER_MAX_CONCURRENT` | `64` | process-wide concurrent provider call limit shared by all tenants and direct/kernel paths; `0` disables |
| `MIYA_PROVIDER_QUEUE_TIMEOUT_MS` | `30000` | maximum wait for a process-wide provider permit |
| `MIYA_MAX_CONCURRENT_ORCHESTRATIONS` | auto: `max(1, provider concurrency / child parallelism)` | process-wide full-orchestration request admission; `0` disables this additional layer without changing per-request agent coverage |
| `MIYA_ORCHESTRATION_QUEUE_TIMEOUT_MS` | `30000` | maximum wait before a full orchestration starts; overload returns `503 orchestration_overloaded` plus `Retry-After` before any partial agent execution |
| `MIYA_PROVIDER_MAX_RETRIES` | `2` | retries after the initial attempt for retryable transport, 408/409/429, and 5xx failures; applies to kernel, direct, streaming setup, and batch paths |
| `MIYA_PROVIDER_RETRY_BASE_MS` | `250` | exponential retry base delay; provider `Retry-After` takes precedence |
| `MIYA_PROVIDER_CIRCUIT_FAILURE_THRESHOLD` | `5` | terminal retryable failures before opening the process-wide provider circuit |
| `MIYA_PROVIDER_CIRCUIT_COOLDOWN_MS` | `30000` | circuit-open cooldown before one half-open probe |
| `MIYA_REQUEST_TIMEOUT_MS` | `3600000` | total kernel request timeout; sized for full high/xhigh slow-model orchestration |
| `MIYA_AGENT_TIMEOUT_MS` | `330000` | timeout for each planner/worker/root/synthesizer/verifier call, slightly above the default upstream HTTP timeout |
| `MIYA_PROVIDER_TIMEOUT_SECS` | `300` | HTTP request timeout for provider adapters and direct passthrough |
| `MIYA_STREAM_HEARTBEAT_SECS` | `10` | SSE comment heartbeat interval while full orchestration is still running; does not alter model output |
| `MIYA_PROVIDER_CONNECT_TIMEOUT_SECS` | `30` | provider connect timeout |
| `MIYA_DATA_DIR` | `.multi-agent-data` | durable object/blob root for files, batches, Message Batches, background Response payloads, and fallback response state |
| `MIYA_MAX_CONCURRENT_JOBS` | `4` | concurrent durable jobs; does not reduce each request's agent coverage |
| `MIYA_BATCH_ITEM_CONCURRENCY` | `8` | requests concurrently dispatched inside one batch, still bounded by tenant/provider admission |
| `MIYA_SEMANTIC_VERIFIER` | `true` in env-built router | enables independent semantic verification after multi-agent synthesis; direct `reasoning.effort=none` remains direct |
| `MIYA_SEMANTIC_MAX_REPAIR_ATTEMPTS` | `2` | maximum repair-and-recheck iterations after a failed semantic verdict |
| `MIYA_PUBLIC_REASONING` | `always` | public multi-agent reasoning summary policy: `always`, `request`, `never`/`strip`; aliases: `MIYA_PUBLIC_REASONING_MODE`, `MULTI_AGENT_PUBLIC_REASONING`, `PUBLIC_REASONING_MODE` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | disabled when unset | enables OTLP/gRPC OpenTelemetry span export; W3C `traceparent` propagation and local tracing remain available |
| `MIYA_OTEL_ENABLED` | `false` | force OTLP initialization when exporter configuration is supplied through standard OTEL variables |
| `RUST_LOG` | `info` | tracing filter for local and OTLP spans |
| `TRAINING_TRACE` | disabled | set to `enabled`, `true`, or `1` to append training samples |
| `TRAINING_TRACE_PATH` | `logs/training-traces.jsonl` | JSONL training sample output path |

## Usage And Telemetry

OpenAI-compatible responses include:

```json
{
  "usage": {
    "prompt_tokens": 223,
    "completion_tokens": 2,
    "total_tokens": 225
  }
}
```

Anthropic-compatible responses include:

```json
{
  "usage": {
    "input_tokens": 223,
    "output_tokens": 2
  }
}
```

The backend also emits compact JSONL records to stdout with `event: "api_usage"`. The Windows launcher redirects these records to `logs\api-server.out.log`. `GET /metrics` exposes Prometheus counters/gauges for HTTP/provider attempts/retries/failures/circuit rejections, durable job lifecycle, orchestration admission limit/active/queued/admitted/queue-timeout/wait time, kernel request/agent timeouts, and active orchestration streams/heartbeat delivery. HTTP requests accept W3C `traceparent`; setting `OTEL_EXPORTER_OTLP_ENDPOINT` enables OTLP/gRPC export.

Telemetry fields include `route`, `model`, `tenant_id`, `request_id`, `conversation_fingerprint`, `reasoning_effort`, `stream`, `batch_index`, `direct_passthrough`, `input_tokens`, `output_tokens`, `total_tokens`, `provider_call_count`, `task_count`, `child_agent_count`, `tool_call_count`, `verification`, and optional context-cache details.

For CLI correlation, send `x-request-id`; the gateway uses that ID in telemetry. The Windows `invoke-miya-api.ps1` script generates one automatically, sends the request, then prints the matching telemetry row from the JSONL log.

Telemetry deliberately does not log raw prompts, final answer text, child-agent artifacts, child-agent tool calls, or hidden thinking. Root provider streaming paths record token usage when the upstream stream exposes a usage event; non-streaming paths use provider response usage directly. Upstream HTTP failures retain provider, status, code, message and retry timing internally; public compatibility errors preserve 429/client status where appropriate and map transport/invalid-response/circuit/admission failures to explicit gateway errors.

## Training Trace Recording

Training trace recording is separate from usage telemetry. It is designed for building your own model training dataset, and is opt-in because it records raw input, output, tool calls, tool observations, and structured orchestration steps.

Enable it:

```bash
TRAINING_TRACE=enabled \
TRAINING_TRACE_PATH=logs/training-traces.jsonl \
cargo run -p api-server
```

Each JSONL line is a training sample in this schema:

```json
{
  "conversations": [
    {"from": "human", "value": "人类指令"},
    {"from": "function_call", "value": "{\"name\":\"lookup\",\"arguments\":{\"key\":\"x\"}}"},
    {"from": "observation", "value": "{\"result\":{\"value\":\"42\"}}"},
    {"from": "gpt", "value": "模型回答"}
  ],
  "system": "系统提示词",
  "tools": "[{\"name\":\"lookup\",\"description\":\"...\"}]"
}
```

For multi-agent orchestration, the recorder also converts bounded sub-agent dispatch into trainable tool-use turns with `spawn_agent` as an internal tool and child-agent outputs as `observation` turns. This records the useful intermediate process without relying on hidden provider reasoning.

Windows commands:

```powershell
.\scripts\windows\start-miya-api.ps1 -TrainingTrace
.\scripts\windows\invoke-miya-api.ps1 -ShowTrainingTrace
.\scripts\windows\watch-miya-training-traces.ps1 -Follow
.\scripts\windows\export-miya-training-dataset.ps1
```

## Examples

### OpenAI Chat Completions

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "local-qwen-model",
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
- reasoning effort to agent coverage mapping
- scoped artifact isolation
- scoped tool ledger behavior
- spawn depth and agent-count rejection
- bounded spawn execution
- encrypted sub-agent state
- OpenAI request normalization
- Anthropic request normalization
- image/audio/file/document/citation/server-tool content normalization and provider preservation
- tool call and tool result compatibility
- legacy OpenAI function compatibility
- OpenAI and Anthropic route responses
- response usage mapping and backend usage telemetry
- SSE formatting and upstream SSE parsing
- official OpenAI multipart Files + asynchronous Batch JSONL lifecycle
- official Anthropic Message Batches create/retrieve/list/cancel/delete/results lifecycle
- durable filesystem reopen and unfinished-job recovery primitives
- Responses background polling and in-flight cancellation
- structured provider errors, 429 retry, circuit-breaker metrics, Prometheus rendering
- semantic verifier artifact coverage and bounded repair/recheck
- SurrealKV context rewind
- context-pack cache reuse
- model-planned high-effort coverage improvement

## License

Miya API is released under the MIT License. See [LICENSE](LICENSE).
