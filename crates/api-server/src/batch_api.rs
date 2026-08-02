use super::*;
use axum::{
    body::to_bytes,
    http::{HeaderValue, header},
};
use futures::{StreamExt, stream};

const DEFAULT_BATCH_ITEM_CONCURRENCY: usize = 8;
const BATCH_RESPONSE_BODY_LIMIT: usize = 128 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub(super) struct OpenAiFilesListQuery {
    after: Option<String>,
    limit: Option<usize>,
    order: Option<String>,
    purpose: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAiBatchCreateRequest {
    completion_window: String,
    endpoint: String,
    input_file_id: String,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    output_expires_after: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAiBatchesListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OpenAiBatchInputLine {
    custom_id: String,
    method: String,
    url: String,
    body: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicBatchCreateRequest {
    requests: Vec<AnthropicBatchInput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AnthropicBatchInput {
    custom_id: String,
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicBatchesListQuery {
    after_id: Option<String>,
    before_id: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug)]
struct BatchDispatchResult {
    status_code: u16,
    body: serde_json::Value,
}

pub(super) async fn metrics(State(state): State<AppState>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(state.metrics.prometheus()))
        .unwrap_or_else(|error| internal_error_response(error.to_string()))
}

pub(super) async fn files_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut purpose = None;
    let mut filename = None;
    let mut file_bytes = None;
    let mut expires_after_seconds = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => {
                return api_error_response(ApiError::InvalidRequest(format!(
                    "invalid multipart file upload: {error}"
                )));
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            filename = field.file_name().map(str::to_string);
            match field.bytes().await {
                Ok(bytes) if bytes.len() <= MAX_OPENAI_FILE_BYTES => {
                    file_bytes = Some(bytes.to_vec());
                }
                Ok(_) => {
                    return api_error_response(ApiError::InvalidRequest(format!(
                        "file exceeds {MAX_OPENAI_FILE_BYTES} bytes"
                    )));
                }
                Err(error) => {
                    return api_error_response(ApiError::InvalidRequest(format!(
                        "failed to read uploaded file: {error}"
                    )));
                }
            }
        } else {
            let text = match field.text().await {
                Ok(text) => text,
                Err(error) => {
                    return api_error_response(ApiError::InvalidRequest(format!(
                        "invalid multipart field {name}: {error}"
                    )));
                }
            };
            match name.as_str() {
                "purpose" => purpose = Some(text),
                "expires_after" => {
                    expires_after_seconds = serde_json::from_str::<serde_json::Value>(&text)
                        .ok()
                        .and_then(|value| {
                            value.get("seconds").and_then(|seconds| seconds.as_u64())
                        });
                }
                "expires_after[seconds]" => {
                    expires_after_seconds = text.parse::<u64>().ok();
                }
                _ => {}
            }
        }
    }

    let purpose = match purpose.filter(|purpose| !purpose.trim().is_empty()) {
        Some(purpose) => purpose,
        None => {
            return api_error_response(ApiError::InvalidRequest(
                "file upload requires purpose".to_string(),
            ));
        }
    };
    if !matches!(
        purpose.as_str(),
        "assistants" | "batch" | "fine-tune" | "vision" | "user_data" | "evals"
    ) {
        return api_error_response(ApiError::InvalidRequest(format!(
            "unsupported file purpose: {purpose}"
        )));
    }
    let filename = match filename.filter(|filename| !filename.trim().is_empty()) {
        Some(filename) => filename,
        None => {
            return api_error_response(ApiError::InvalidRequest(
                "file upload requires a filename".to_string(),
            ));
        }
    };
    let file_bytes = match file_bytes {
        Some(bytes) => bytes,
        None => {
            return api_error_response(ApiError::InvalidRequest(
                "file upload requires file content".to_string(),
            ));
        }
    };
    if purpose == "batch" && file_bytes.len() > MAX_OPENAI_BATCH_FILE_BYTES {
        return api_error_response(ApiError::InvalidRequest(format!(
            "batch files must not exceed {MAX_OPENAI_BATCH_FILE_BYTES} bytes"
        )));
    }

    let created_at = unix_timestamp_secs();
    let id = format!("file-{}", Uuid::new_v4().simple());
    let expires_at = expires_after_seconds
        .map(|seconds| created_at.saturating_add(seconds))
        .or_else(|| (purpose == "batch").then(|| created_at.saturating_add(30 * 24 * 60 * 60)));
    let file = StoredOpenAiFile {
        tenant_id: tenant_id.as_ref().to_string(),
        id: id.clone(),
        bytes: file_bytes.len() as u64,
        created_at,
        filename,
        purpose,
        status: "processed".to_string(),
        expires_at,
        status_details: None,
    };
    if let Err(error) = state
        .durable
        .put_blob(
            OPENAI_FILE_BLOBS_NAMESPACE,
            tenant_id.as_ref(),
            &id,
            file_bytes,
        )
        .await
    {
        return internal_error_response(error);
    }
    if let Err(error) = state
        .durable
        .put_json(OPENAI_FILES_NAMESPACE, tenant_id.as_ref(), &id, &file)
        .await
    {
        let _ = state
            .durable
            .delete_blob(OPENAI_FILE_BLOBS_NAMESPACE, tenant_id.as_ref(), &id)
            .await;
        return internal_error_response(error);
    }

    Json(openai_file_json(&file)).into_response()
}

pub(super) async fn files_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(file_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    match state
        .durable
        .get_json::<StoredOpenAiFile>(OPENAI_FILES_NAMESPACE, tenant_id.as_ref(), &file_id)
        .await
    {
        Ok(Some(file)) => Json(openai_file_json(&file)).into_response(),
        Ok(None) => not_found_response(format!("file not found: {file_id}")),
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn files_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(file_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let file = match state
        .durable
        .get_json::<StoredOpenAiFile>(OPENAI_FILES_NAMESPACE, tenant_id.as_ref(), &file_id)
        .await
    {
        Ok(Some(file)) => file,
        Ok(None) => return not_found_response(format!("file not found: {file_id}")),
        Err(error) => return internal_error_response(error),
    };
    match state
        .durable
        .get_blob(OPENAI_FILE_BLOBS_NAMESPACE, tenant_id.as_ref(), &file_id)
        .await
    {
        Ok(Some(bytes)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(
                header::CONTENT_DISPOSITION,
                format!(
                    "attachment; filename=\"{}\"",
                    safe_header_filename(&file.filename)
                ),
            )
            .body(Body::from(bytes))
            .unwrap_or_else(|error| internal_error_response(error.to_string())),
        Ok(None) => internal_error_response(format!("file content missing: {file_id}")),
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn files_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<OpenAiFilesListQuery>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut files = match state
        .durable
        .list_json::<StoredOpenAiFile>(OPENAI_FILES_NAMESPACE, tenant_id.as_ref())
        .await
    {
        Ok(files) => files,
        Err(error) => return internal_error_response(error),
    };
    if let Some(purpose) = query.purpose.as_deref() {
        files.retain(|file| file.purpose == purpose);
    }
    let ascending = query.order.as_deref() == Some("asc");
    files.sort_by(|left, right| {
        let ordering = left
            .created_at
            .cmp(&right.created_at)
            .then_with(|| left.id.cmp(&right.id));
        if ascending {
            ordering
        } else {
            ordering.reverse()
        }
    });
    let limit = query.limit.unwrap_or(10_000).clamp(1, 10_000);
    let (page, has_more) = page_after(files, query.after.as_deref(), limit);
    let first_id = page.first().map(|file| file.id.clone());
    let last_id = page.last().map(|file| file.id.clone());
    Json(serde_json::json!({
        "object": "list",
        "data": page.iter().map(openai_file_json).collect::<Vec<_>>(),
        "has_more": has_more,
        "first_id": first_id,
        "last_id": last_id
    }))
    .into_response()
}

pub(super) async fn files_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(file_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let batches = match state
        .durable
        .list_json::<OpenAiBatchRecord>(OPENAI_BATCHES_NAMESPACE, tenant_id.as_ref())
        .await
    {
        Ok(batches) => batches,
        Err(error) => return internal_error_response(error),
    };
    if batches.iter().any(|batch| {
        batch.input_file_id == file_id
            && matches!(
                batch.status.as_str(),
                "validating" | "in_progress" | "finalizing" | "cancelling"
            )
    }) {
        return conflict_response(format!("file {file_id} is still used by an active batch"));
    }
    match state
        .durable
        .delete_json(OPENAI_FILES_NAMESPACE, tenant_id.as_ref(), &file_id)
        .await
    {
        Ok(true) => {
            let _ = state
                .durable
                .delete_blob(OPENAI_FILE_BLOBS_NAMESPACE, tenant_id.as_ref(), &file_id)
                .await;
            Json(serde_json::json!({
                "id": file_id,
                "object": "file",
                "deleted": true
            }))
            .into_response()
        }
        Ok(false) => not_found_response(format!("file not found: {file_id}")),
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn openai_batches_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<OpenAiBatchCreateRequest>,
) -> Response {
    if request.completion_window != "24h" {
        return api_error_response(ApiError::InvalidRequest(
            "completion_window must be 24h".to_string(),
        ));
    }
    if !matches!(
        request.endpoint.as_str(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
    ) {
        return api_error_response(ApiError::InvalidRequest(format!(
            "batch endpoint {} is not implemented by this gateway",
            request.endpoint
        )));
    }
    if !request.output_expires_after.is_null()
        && request
            .output_expires_after
            .get("anchor")
            .and_then(serde_json::Value::as_str)
            != Some("created_at")
    {
        return api_error_response(ApiError::InvalidRequest(
            "output_expires_after.anchor must be created_at".to_string(),
        ));
    }
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let input_file = match state
        .durable
        .get_json::<StoredOpenAiFile>(
            OPENAI_FILES_NAMESPACE,
            tenant_id.as_ref(),
            &request.input_file_id,
        )
        .await
    {
        Ok(Some(file)) => file,
        Ok(None) => {
            return api_error_response(ApiError::InvalidRequest(format!(
                "input_file_id not found: {}",
                request.input_file_id
            )));
        }
        Err(error) => return internal_error_response(error),
    };
    if input_file.purpose != "batch" {
        return api_error_response(ApiError::InvalidRequest(
            "input_file_id must reference a file with purpose=batch".to_string(),
        ));
    }

    let created_at = unix_timestamp_secs();
    let id = format!("batch_{}", Uuid::new_v4().simple());
    let output_ttl = request
        .output_expires_after
        .get("seconds")
        .and_then(serde_json::Value::as_u64);
    let batch = OpenAiBatchRecord {
        tenant_id: tenant_id.as_ref().to_string(),
        id: id.clone(),
        completion_window: request.completion_window,
        created_at,
        endpoint: request.endpoint,
        input_file_id: request.input_file_id,
        status: "validating".to_string(),
        cancelled_at: None,
        cancelling_at: None,
        completed_at: None,
        error_file_id: None,
        errors: None,
        expired_at: None,
        expires_at: output_ttl.map(|ttl| created_at.saturating_add(ttl)),
        failed_at: None,
        finalizing_at: None,
        in_progress_at: None,
        metadata: if request.metadata.is_null() {
            serde_json::json!({})
        } else {
            request.metadata
        },
        output_file_id: None,
        request_counts: OpenAiBatchRequestCounts::default(),
        usage: ProviderUsage::default(),
        cancel_requested: false,
    };
    if let Err(error) = state
        .durable
        .put_json(OPENAI_BATCHES_NAMESPACE, tenant_id.as_ref(), &id, &batch)
        .await
    {
        return internal_error_response(error);
    }
    spawn_openai_batch(state, batch.clone());
    Json(openai_batch_json(&batch)).into_response()
}

pub(super) async fn openai_batches_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    match state
        .durable
        .get_json::<OpenAiBatchRecord>(OPENAI_BATCHES_NAMESPACE, tenant_id.as_ref(), &batch_id)
        .await
    {
        Ok(Some(batch)) => Json(openai_batch_json(&batch)).into_response(),
        Ok(None) => not_found_response(format!("batch not found: {batch_id}")),
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn openai_batches_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<OpenAiBatchesListQuery>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut batches = match state
        .durable
        .list_json::<OpenAiBatchRecord>(OPENAI_BATCHES_NAMESPACE, tenant_id.as_ref())
        .await
    {
        Ok(batches) => batches,
        Err(error) => return internal_error_response(error),
    };
    batches.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| right.id.cmp(&left.id))
    });
    let limit = query.limit.unwrap_or(20).clamp(1, 100);
    let (page, has_more) = page_after(batches, query.after.as_deref(), limit);
    let first_id = page.first().map(|batch| batch.id.clone());
    let last_id = page.last().map(|batch| batch.id.clone());
    Json(serde_json::json!({
        "object": "list",
        "data": page.iter().map(openai_batch_json).collect::<Vec<_>>(),
        "has_more": has_more,
        "first_id": first_id,
        "last_id": last_id
    }))
    .into_response()
}

pub(super) async fn openai_batches_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut batch = match state
        .durable
        .get_json::<OpenAiBatchRecord>(OPENAI_BATCHES_NAMESPACE, tenant_id.as_ref(), &batch_id)
        .await
    {
        Ok(Some(batch)) => batch,
        Ok(None) => return not_found_response(format!("batch not found: {batch_id}")),
        Err(error) => return internal_error_response(error),
    };
    if matches!(
        batch.status.as_str(),
        "completed" | "failed" | "cancelled" | "expired"
    ) {
        return api_error_response(ApiError::InvalidRequest(format!(
            "batch {batch_id} is already terminal"
        )));
    }
    batch.cancel_requested = true;
    batch.status = "cancelling".to_string();
    batch.cancelling_at = Some(unix_timestamp_secs());
    if let Err(error) = state
        .durable
        .put_json(
            OPENAI_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
            &batch,
        )
        .await
    {
        return internal_error_response(error);
    }
    state.jobs.cancel(&openai_batch_job_key(&batch));
    Json(openai_batch_json(&batch)).into_response()
}

pub(super) async fn anthropic_batches_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AnthropicBatchCreateRequest>,
) -> Response {
    if request.requests.is_empty() || request.requests.len() > MAX_ANTHROPIC_BATCH_REQUESTS {
        return api_error_response(ApiError::InvalidRequest(format!(
            "requests must contain between 1 and {MAX_ANTHROPIC_BATCH_REQUESTS} items"
        )));
    }
    let mut custom_ids = HashSet::new();
    for item in &request.requests {
        if item.custom_id.trim().is_empty() || !custom_ids.insert(item.custom_id.clone()) {
            return api_error_response(ApiError::InvalidRequest(
                "each Message Batch custom_id must be non-empty and unique".to_string(),
            ));
        }
        if item
            .params
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            return api_error_response(ApiError::InvalidRequest(
                "Message Batch requests cannot stream".to_string(),
            ));
        }
    }

    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let created_at = unix_timestamp_secs();
    let id = format!("msgbatch_{}", Uuid::new_v4().simple());
    let input = match serde_json::to_vec(&request.requests) {
        Ok(input) => input,
        Err(error) => return internal_error_response(error.to_string()),
    };
    if let Err(error) = state
        .durable
        .put_blob(
            ANTHROPIC_BATCH_INPUTS_NAMESPACE,
            tenant_id.as_ref(),
            &id,
            input,
        )
        .await
    {
        return internal_error_response(error);
    }
    let batch = AnthropicBatchRecord {
        tenant_id: tenant_id.as_ref().to_string(),
        id: id.clone(),
        created_at,
        expires_at: created_at.saturating_add(24 * 60 * 60),
        ended_at: None,
        cancel_initiated_at: None,
        archived_at: None,
        processing_status: "in_progress".to_string(),
        request_counts: AnthropicBatchRequestCounts {
            processing: request.requests.len() as u64,
            ..AnthropicBatchRequestCounts::default()
        },
        cancel_requested: false,
    };
    if let Err(error) = state
        .durable
        .put_json(ANTHROPIC_BATCHES_NAMESPACE, tenant_id.as_ref(), &id, &batch)
        .await
    {
        let _ = state
            .durable
            .delete_blob(ANTHROPIC_BATCH_INPUTS_NAMESPACE, tenant_id.as_ref(), &id)
            .await;
        return internal_error_response(error);
    }
    spawn_anthropic_batch(state, batch.clone());
    Json(anthropic_batch_json(&batch)).into_response()
}

pub(super) async fn anthropic_batches_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    match state
        .durable
        .get_json::<AnthropicBatchRecord>(
            ANTHROPIC_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await
    {
        Ok(Some(batch)) => Json(anthropic_batch_json(&batch)).into_response(),
        Ok(None) => not_found_response(format!("message batch not found: {batch_id}")),
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn anthropic_batches_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnthropicBatchesListQuery>,
) -> Response {
    if query.after_id.is_some() && query.before_id.is_some() {
        return api_error_response(ApiError::InvalidRequest(
            "after_id and before_id cannot be used together".to_string(),
        ));
    }
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut batches = match state
        .durable
        .list_json::<AnthropicBatchRecord>(ANTHROPIC_BATCHES_NAMESPACE, tenant_id.as_ref())
        .await
    {
        Ok(batches) => batches,
        Err(error) => return internal_error_response(error),
    };
    batches.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| right.id.cmp(&left.id))
    });
    if let Some(before_id) = query.before_id.as_deref()
        && let Some(position) = batches.iter().position(|batch| batch.id == before_id)
    {
        batches.truncate(position);
    }
    let limit = query.limit.unwrap_or(20).clamp(1, 1000);
    let (page, has_more) = page_after(batches, query.after_id.as_deref(), limit);
    let first_id = page.first().map(|batch| batch.id.clone());
    let last_id = page.last().map(|batch| batch.id.clone());
    Json(serde_json::json!({
        "data": page.iter().map(anthropic_batch_json).collect::<Vec<_>>(),
        "has_more": has_more,
        "first_id": first_id,
        "last_id": last_id
    }))
    .into_response()
}

pub(super) async fn anthropic_batches_results(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let batch = match state
        .durable
        .get_json::<AnthropicBatchRecord>(
            ANTHROPIC_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await
    {
        Ok(Some(batch)) => batch,
        Ok(None) => return not_found_response(format!("message batch not found: {batch_id}")),
        Err(error) => return internal_error_response(error),
    };
    if batch.processing_status != "ended" {
        return conflict_response(format!("message batch {batch_id} has not ended"));
    }
    match state
        .durable
        .get_blob(
            ANTHROPIC_BATCH_RESULTS_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await
    {
        Ok(Some(bytes)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from(bytes))
            .unwrap_or_else(|error| internal_error_response(error.to_string())),
        Ok(None) => {
            internal_error_response(format!("results missing for message batch {batch_id}"))
        }
        Err(error) => internal_error_response(error),
    }
}

pub(super) async fn anthropic_batches_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let mut batch = match state
        .durable
        .get_json::<AnthropicBatchRecord>(
            ANTHROPIC_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await
    {
        Ok(Some(batch)) => batch,
        Ok(None) => return not_found_response(format!("message batch not found: {batch_id}")),
        Err(error) => return internal_error_response(error),
    };
    if batch.processing_status == "ended" {
        return api_error_response(ApiError::InvalidRequest(format!(
            "message batch {batch_id} has already ended"
        )));
    }
    batch.cancel_requested = true;
    batch.processing_status = "canceling".to_string();
    batch.cancel_initiated_at = Some(unix_timestamp_secs());
    if let Err(error) = state
        .durable
        .put_json(
            ANTHROPIC_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
            &batch,
        )
        .await
    {
        return internal_error_response(error);
    }
    state.jobs.cancel(&anthropic_batch_job_key(&batch));
    Json(anthropic_batch_json(&batch)).into_response()
}

pub(super) async fn anthropic_batches_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let tenant_id = RequestContext::from_headers(&headers).tenant_id();
    let batch = match state
        .durable
        .get_json::<AnthropicBatchRecord>(
            ANTHROPIC_BATCHES_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await
    {
        Ok(Some(batch)) => batch,
        Ok(None) => return not_found_response(format!("message batch not found: {batch_id}")),
        Err(error) => return internal_error_response(error),
    };
    if batch.processing_status != "ended" {
        return conflict_response(
            "Message Batches can only be deleted after processing has ended".to_string(),
        );
    }
    if let Err(error) = state
        .durable
        .delete_json(ANTHROPIC_BATCHES_NAMESPACE, tenant_id.as_ref(), &batch_id)
        .await
    {
        return internal_error_response(error);
    }
    let _ = state
        .durable
        .delete_blob(
            ANTHROPIC_BATCH_INPUTS_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await;
    let _ = state
        .durable
        .delete_blob(
            ANTHROPIC_BATCH_RESULTS_NAMESPACE,
            tenant_id.as_ref(),
            &batch_id,
        )
        .await;
    Json(serde_json::json!({
        "id": batch_id,
        "type": "message_batch_deleted"
    }))
    .into_response()
}

pub(super) fn recover_durable_jobs(state: AppState) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    handle.spawn(async move {
        if let Ok(batches) = state
            .durable
            .list_all_json::<OpenAiBatchRecord>(OPENAI_BATCHES_NAMESPACE)
            .await
        {
            for batch in batches {
                if matches!(
                    batch.status.as_str(),
                    "validating" | "in_progress" | "finalizing" | "cancelling"
                ) {
                    spawn_openai_batch(state.clone(), batch);
                }
            }
        }
        if let Ok(batches) = state
            .durable
            .list_all_json::<AnthropicBatchRecord>(ANTHROPIC_BATCHES_NAMESPACE)
            .await
        {
            for batch in batches {
                if matches!(
                    batch.processing_status.as_str(),
                    "in_progress" | "canceling"
                ) {
                    spawn_anthropic_batch(state.clone(), batch);
                }
            }
        }
        recover_background_response_jobs(state).await;
    });
}

fn spawn_openai_batch(state: AppState, batch: OpenAiBatchRecord) {
    let key = openai_batch_job_key(&batch);
    let runtime = state.jobs.clone();
    runtime.spawn(key, move |cancellation| async move {
        if let Err(error) = run_openai_batch(state.clone(), batch.clone(), cancellation).await {
            fail_openai_batch(&state, batch, error).await;
        }
    });
}

async fn run_openai_batch(
    state: AppState,
    mut batch: OpenAiBatchRecord,
    cancellation: CancellationToken,
) -> Result<(), String> {
    let tenant_id = batch.tenant_id.clone();
    if batch.cancel_requested || cancellation.is_cancelled() {
        return cancel_openai_batch(&state, batch, Vec::new(), Vec::new()).await;
    }
    let file = state
        .durable
        .get_json::<StoredOpenAiFile>(OPENAI_FILES_NAMESPACE, &tenant_id, &batch.input_file_id)
        .await?
        .ok_or_else(|| format!("input file not found: {}", batch.input_file_id))?;
    if file.purpose != "batch" {
        return Err("input file purpose must be batch".to_string());
    }
    let bytes = state
        .durable
        .get_blob(
            OPENAI_FILE_BLOBS_NAMESPACE,
            &tenant_id,
            &batch.input_file_id,
        )
        .await?
        .ok_or_else(|| "input file content is missing".to_string())?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("batch input must be UTF-8 JSONL: {error}"))?;
    let mut inputs = Vec::new();
    let mut custom_ids = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if inputs.len() >= MAX_OPENAI_BATCH_REQUESTS {
            return Err(format!(
                "batch input exceeds {MAX_OPENAI_BATCH_REQUESTS} requests"
            ));
        }
        let input = serde_json::from_str::<OpenAiBatchInputLine>(line)
            .map_err(|error| format!("invalid JSONL at line {}: {error}", index + 1))?;
        if input.custom_id.trim().is_empty() || !custom_ids.insert(input.custom_id.clone()) {
            return Err(format!(
                "custom_id must be non-empty and unique at line {}",
                index + 1
            ));
        }
        if !input.method.eq_ignore_ascii_case("POST") {
            return Err(format!("batch method must be POST at line {}", index + 1));
        }
        if input.url != batch.endpoint {
            return Err(format!(
                "batch URL {} does not match endpoint {} at line {}",
                input.url,
                batch.endpoint,
                index + 1
            ));
        }
        inputs.push(input);
    }
    if inputs.is_empty() {
        return Err("batch input file contains no requests".to_string());
    }

    batch.status = "in_progress".to_string();
    batch.in_progress_at = Some(unix_timestamp_secs());
    batch.request_counts.total = inputs.len() as u64;
    state
        .durable
        .put_json(OPENAI_BATCHES_NAMESPACE, &tenant_id, &batch.id, &batch)
        .await?;

    let item_concurrency = batch_item_concurrency();
    let endpoint = batch.endpoint.clone();
    let results = stream::iter(inputs.into_iter().map(|input| {
        let state = state.clone();
        let tenant_id = tenant_id.clone();
        let endpoint = endpoint.clone();
        let cancellation = cancellation.clone();
        async move {
            let custom_id = input.custom_id.clone();
            tokio::select! {
                _ = cancellation.cancelled() => (custom_id, None),
                result = dispatch_openai_batch_item(&state, &tenant_id, &endpoint, input.body) => {
                    (custom_id, Some(result))
                }
            }
        }
    }))
    .buffer_unordered(item_concurrency)
    .collect::<Vec<_>>()
    .await;

    let mut output_lines = Vec::new();
    let mut error_lines = Vec::new();
    let mut usage = ProviderUsage::default();
    let mut cancelled = 0_u64;
    for (custom_id, result) in results {
        match result {
            None => cancelled = cancelled.saturating_add(1),
            Some(Ok(result)) => {
                let request_id = format!("batch_req_{}", Uuid::new_v4().simple());
                let line = serde_json::json!({
                    "id": request_id,
                    "custom_id": custom_id,
                    "response": {
                        "status_code": result.status_code,
                        "request_id": request_id,
                        "body": result.body
                    },
                    "error": null
                });
                if (200..300).contains(&result.status_code) {
                    batch.request_counts.completed =
                        batch.request_counts.completed.saturating_add(1);
                    accumulate_json_usage(&mut usage, &line["response"]["body"]);
                    output_lines.push(line);
                } else {
                    batch.request_counts.failed = batch.request_counts.failed.saturating_add(1);
                    error_lines.push(line);
                }
            }
            Some(Err(error)) => {
                batch.request_counts.failed = batch.request_counts.failed.saturating_add(1);
                error_lines.push(serde_json::json!({
                    "id": format!("batch_req_{}", Uuid::new_v4().simple()),
                    "custom_id": custom_id,
                    "response": null,
                    "error": {
                        "code": "batch_dispatch_error",
                        "message": error
                    }
                }));
            }
        }
    }
    batch.request_counts.failed = batch.request_counts.failed.saturating_add(cancelled);
    batch.usage = usage;
    if cancellation.is_cancelled() || batch.cancel_requested || cancelled > 0 {
        return cancel_openai_batch(&state, batch, output_lines, error_lines).await;
    }

    batch.status = "finalizing".to_string();
    batch.finalizing_at = Some(unix_timestamp_secs());
    state
        .durable
        .put_json(OPENAI_BATCHES_NAMESPACE, &tenant_id, &batch.id, &batch)
        .await?;
    batch.output_file_id =
        write_openai_batch_output_file(&state, &tenant_id, &batch.id, "output", output_lines)
            .await?;
    batch.error_file_id =
        write_openai_batch_output_file(&state, &tenant_id, &batch.id, "errors", error_lines)
            .await?;
    batch.status = "completed".to_string();
    batch.completed_at = Some(unix_timestamp_secs());
    state
        .durable
        .put_json(OPENAI_BATCHES_NAMESPACE, &tenant_id, &batch.id, &batch)
        .await
}

async fn cancel_openai_batch(
    state: &AppState,
    mut batch: OpenAiBatchRecord,
    output_lines: Vec<serde_json::Value>,
    error_lines: Vec<serde_json::Value>,
) -> Result<(), String> {
    let tenant_id = batch.tenant_id.clone();
    batch.output_file_id =
        write_openai_batch_output_file(state, &tenant_id, &batch.id, "output", output_lines)
            .await?;
    batch.error_file_id =
        write_openai_batch_output_file(state, &tenant_id, &batch.id, "errors", error_lines).await?;
    batch.status = "cancelled".to_string();
    batch.cancelled_at = Some(unix_timestamp_secs());
    state
        .durable
        .put_json(OPENAI_BATCHES_NAMESPACE, &tenant_id, &batch.id, &batch)
        .await
}

async fn fail_openai_batch(state: &AppState, mut batch: OpenAiBatchRecord, error: String) {
    let now = unix_timestamp_secs();
    batch.status = "failed".to_string();
    batch.failed_at = Some(now);
    batch.errors = Some(serde_json::json!({
        "object": "list",
        "data": [{
            "code": "invalid_batch_file",
            "message": error,
            "param": "input_file_id",
            "line": null
        }]
    }));
    if let Ok(file_id) = write_openai_batch_output_file(
        state,
        &batch.tenant_id,
        &batch.id,
        "validation-errors",
        vec![serde_json::json!({
            "error": batch.errors.as_ref().and_then(|errors| errors.get("data")).and_then(|data| data.get(0)).cloned()
        })],
    )
    .await
    {
        batch.error_file_id = file_id;
    }
    let _ = state
        .durable
        .put_json(
            OPENAI_BATCHES_NAMESPACE,
            &batch.tenant_id,
            &batch.id,
            &batch,
        )
        .await;
}

fn spawn_anthropic_batch(state: AppState, batch: AnthropicBatchRecord) {
    let key = anthropic_batch_job_key(&batch);
    let runtime = state.jobs.clone();
    runtime.spawn(key, move |cancellation| async move {
        if let Err(error) = run_anthropic_batch(state.clone(), batch.clone(), cancellation).await {
            fail_anthropic_batch(&state, batch, error).await;
        }
    });
}

async fn run_anthropic_batch(
    state: AppState,
    mut batch: AnthropicBatchRecord,
    cancellation: CancellationToken,
) -> Result<(), String> {
    let bytes = state
        .durable
        .get_blob(
            ANTHROPIC_BATCH_INPUTS_NAMESPACE,
            &batch.tenant_id,
            &batch.id,
        )
        .await?
        .ok_or_else(|| "Message Batch input is missing".to_string())?;
    let inputs = serde_json::from_slice::<Vec<AnthropicBatchInput>>(&bytes)
        .map_err(|error| format!("invalid persisted Message Batch input: {error}"))?;
    if batch.cancel_requested || cancellation.is_cancelled() {
        return finish_cancelled_anthropic_batch(&state, batch, &inputs).await;
    }

    let item_concurrency = batch_item_concurrency();
    let tenant_id = batch.tenant_id.clone();
    let results = stream::iter(inputs.into_iter().map(|input| {
        let state = state.clone();
        let tenant_id = tenant_id.clone();
        let cancellation = cancellation.clone();
        async move {
            let custom_id = input.custom_id.clone();
            tokio::select! {
                _ = cancellation.cancelled() => (custom_id, None),
                result = dispatch_anthropic_batch_item(&state, &tenant_id, input.params) => {
                    (custom_id, Some(result))
                }
            }
        }
    }))
    .buffer_unordered(item_concurrency)
    .collect::<Vec<_>>()
    .await;

    let mut lines = Vec::new();
    batch.request_counts = AnthropicBatchRequestCounts::default();
    for (custom_id, result) in results {
        let result = match result {
            None => {
                batch.request_counts.canceled = batch.request_counts.canceled.saturating_add(1);
                serde_json::json!({"type": "canceled"})
            }
            Some(Ok(result)) if (200..300).contains(&result.status_code) => {
                batch.request_counts.succeeded = batch.request_counts.succeeded.saturating_add(1);
                serde_json::json!({"type": "succeeded", "message": result.body})
            }
            Some(Ok(result)) => {
                batch.request_counts.errored = batch.request_counts.errored.saturating_add(1);
                serde_json::json!({
                    "type": "errored",
                    "error": result.body.get("error").cloned().unwrap_or(result.body)
                })
            }
            Some(Err(error)) => {
                batch.request_counts.errored = batch.request_counts.errored.saturating_add(1);
                serde_json::json!({
                    "type": "errored",
                    "error": {
                        "type": "api_error",
                        "message": error
                    }
                })
            }
        };
        lines.push(serde_json::json!({"custom_id": custom_id, "result": result}));
    }
    write_jsonl_blob(
        &state,
        ANTHROPIC_BATCH_RESULTS_NAMESPACE,
        &batch.tenant_id,
        &batch.id,
        lines,
    )
    .await?;
    batch.processing_status = "ended".to_string();
    batch.ended_at = Some(unix_timestamp_secs());
    batch.request_counts.processing = 0;
    state
        .durable
        .put_json(
            ANTHROPIC_BATCHES_NAMESPACE,
            &batch.tenant_id,
            &batch.id,
            &batch,
        )
        .await
}

async fn finish_cancelled_anthropic_batch(
    state: &AppState,
    mut batch: AnthropicBatchRecord,
    inputs: &[AnthropicBatchInput],
) -> Result<(), String> {
    let lines = inputs
        .iter()
        .map(|input| {
            serde_json::json!({
                "custom_id": input.custom_id,
                "result": {"type": "canceled"}
            })
        })
        .collect::<Vec<_>>();
    write_jsonl_blob(
        state,
        ANTHROPIC_BATCH_RESULTS_NAMESPACE,
        &batch.tenant_id,
        &batch.id,
        lines,
    )
    .await?;
    batch.processing_status = "ended".to_string();
    batch.ended_at = Some(unix_timestamp_secs());
    batch.request_counts = AnthropicBatchRequestCounts {
        canceled: inputs.len() as u64,
        ..AnthropicBatchRequestCounts::default()
    };
    state
        .durable
        .put_json(
            ANTHROPIC_BATCHES_NAMESPACE,
            &batch.tenant_id,
            &batch.id,
            &batch,
        )
        .await
}

async fn fail_anthropic_batch(state: &AppState, mut batch: AnthropicBatchRecord, error: String) {
    let line = serde_json::json!({
        "custom_id": "_batch",
        "result": {
            "type": "errored",
            "error": {"type": "api_error", "message": error}
        }
    });
    let _ = write_jsonl_blob(
        state,
        ANTHROPIC_BATCH_RESULTS_NAMESPACE,
        &batch.tenant_id,
        &batch.id,
        vec![line],
    )
    .await;
    batch.processing_status = "ended".to_string();
    batch.ended_at = Some(unix_timestamp_secs());
    batch.request_counts.processing = 0;
    batch.request_counts.errored = batch.request_counts.errored.max(1);
    let _ = state
        .durable
        .put_json(
            ANTHROPIC_BATCHES_NAMESPACE,
            &batch.tenant_id,
            &batch.id,
            &batch,
        )
        .await;
}

async fn dispatch_openai_batch_item(
    state: &AppState,
    tenant_id: &str,
    endpoint: &str,
    mut body: serde_json::Value,
) -> Result<BatchDispatchResult, String> {
    if let Some(object) = body.as_object_mut() {
        object.insert("stream".to_string(), serde_json::Value::Bool(false));
        enforce_batch_tenant(object, tenant_id);
        if endpoint == "/v1/responses" {
            object.insert("background".to_string(), serde_json::Value::Bool(false));
        }
    }
    let headers = tenant_headers(tenant_id)?;
    let response = match endpoint {
        "/v1/chat/completions" => chat_completions(State(state.clone()), headers, Json(body)).await,
        "/v1/completions" => completions(State(state.clone()), headers, Json(body)).await,
        "/v1/responses" => responses(State(state.clone()), headers, Json(body)).await,
        _ => return Err(format!("unsupported batch endpoint: {endpoint}")),
    };
    response_to_batch_result(response).await
}

async fn dispatch_anthropic_batch_item(
    state: &AppState,
    tenant_id: &str,
    mut body: serde_json::Value,
) -> Result<BatchDispatchResult, String> {
    if let Some(object) = body.as_object_mut() {
        object.insert("stream".to_string(), serde_json::Value::Bool(false));
        enforce_batch_tenant(object, tenant_id);
    }
    let headers = tenant_headers(tenant_id)?;
    response_to_batch_result(messages(State(state.clone()), headers, Json(body)).await).await
}

fn enforce_batch_tenant(object: &mut serde_json::Map<String, serde_json::Value>, tenant_id: &str) {
    let metadata = object
        .entry("metadata".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if !metadata.is_object() {
        *metadata = serde_json::json!({});
    }
    metadata["tenant_id"] = serde_json::Value::String(tenant_id.to_string());
}

async fn response_to_batch_result(response: Response) -> Result<BatchDispatchResult, String> {
    let status_code = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), BATCH_RESPONSE_BODY_LIMIT)
        .await
        .map_err(|error| error.to_string())?;
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"body": String::from_utf8_lossy(&bytes)}));
    Ok(BatchDispatchResult { status_code, body })
}

pub(super) fn tenant_headers_for_background(tenant_id: &str) -> Result<HeaderMap, String> {
    tenant_headers(tenant_id)
}

fn tenant_headers(tenant_id: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-tenant-id",
        HeaderValue::from_str(tenant_id).map_err(|error| error.to_string())?,
    );
    Ok(headers)
}

async fn write_openai_batch_output_file(
    state: &AppState,
    tenant_id: &str,
    batch_id: &str,
    kind: &str,
    lines: Vec<serde_json::Value>,
) -> Result<Option<String>, String> {
    if lines.is_empty() {
        return Ok(None);
    }
    let id = format!("file-{}", Uuid::new_v4().simple());
    let bytes = jsonl_bytes(lines)?;
    let file = StoredOpenAiFile {
        tenant_id: tenant_id.to_string(),
        id: id.clone(),
        bytes: bytes.len() as u64,
        created_at: unix_timestamp_secs(),
        filename: format!("{batch_id}-{kind}.jsonl"),
        purpose: "batch_output".to_string(),
        status: "processed".to_string(),
        expires_at: None,
        status_details: None,
    };
    state
        .durable
        .put_blob(OPENAI_FILE_BLOBS_NAMESPACE, tenant_id, &id, bytes)
        .await?;
    state
        .durable
        .put_json(OPENAI_FILES_NAMESPACE, tenant_id, &id, &file)
        .await?;
    Ok(Some(id))
}

async fn write_jsonl_blob(
    state: &AppState,
    namespace: &str,
    tenant_id: &str,
    id: &str,
    lines: Vec<serde_json::Value>,
) -> Result<(), String> {
    state
        .durable
        .put_blob(namespace, tenant_id, id, jsonl_bytes(lines)?)
        .await
}

fn jsonl_bytes(lines: Vec<serde_json::Value>) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for line in lines {
        serde_json::to_writer(&mut bytes, &line).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn openai_file_json(file: &StoredOpenAiFile) -> serde_json::Value {
    serde_json::json!({
        "id": file.id,
        "object": "file",
        "bytes": file.bytes,
        "created_at": file.created_at,
        "filename": file.filename,
        "purpose": file.purpose,
        "status": file.status,
        "expires_at": file.expires_at,
        "status_details": file.status_details
    })
}

fn openai_batch_json(batch: &OpenAiBatchRecord) -> serde_json::Value {
    serde_json::json!({
        "id": batch.id,
        "object": "batch",
        "endpoint": batch.endpoint,
        "errors": batch.errors,
        "input_file_id": batch.input_file_id,
        "completion_window": batch.completion_window,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "created_at": batch.created_at,
        "in_progress_at": batch.in_progress_at,
        "expires_at": batch.expires_at,
        "finalizing_at": batch.finalizing_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "cancelling_at": batch.cancelling_at,
        "cancelled_at": batch.cancelled_at,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed
        },
        "metadata": batch.metadata,
        "usage": {
            "input_tokens": batch.usage.input_tokens,
            "output_tokens": batch.usage.output_tokens,
            "total_tokens": batch.usage.input_tokens.saturating_add(batch.usage.output_tokens)
        }
    })
}

fn anthropic_batch_json(batch: &AnthropicBatchRecord) -> serde_json::Value {
    serde_json::json!({
        "id": batch.id,
        "type": "message_batch",
        "processing_status": batch.processing_status,
        "request_counts": {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired
        },
        "ended_at": batch.ended_at.map(rfc3339_timestamp),
        "created_at": rfc3339_timestamp(batch.created_at),
        "expires_at": rfc3339_timestamp(batch.expires_at),
        "archived_at": batch.archived_at.map(rfc3339_timestamp),
        "cancel_initiated_at": batch.cancel_initiated_at.map(rfc3339_timestamp),
        "results_url": (batch.processing_status == "ended")
            .then(|| format!("/v1/messages/batches/{}/results", batch.id))
    })
}

fn page_after<T>(items: Vec<T>, after: Option<&str>, limit: usize) -> (Vec<T>, bool)
where
    T: ObjectId,
{
    let start = after
        .and_then(|after| items.iter().position(|item| item.object_id() == after))
        .map(|position| position + 1)
        .unwrap_or(0);
    let has_more = start.saturating_add(limit) < items.len();
    (
        items.into_iter().skip(start).take(limit).collect(),
        has_more,
    )
}

trait ObjectId {
    fn object_id(&self) -> &str;
}

impl ObjectId for StoredOpenAiFile {
    fn object_id(&self) -> &str {
        &self.id
    }
}

impl ObjectId for OpenAiBatchRecord {
    fn object_id(&self) -> &str {
        &self.id
    }
}

impl ObjectId for AnthropicBatchRecord {
    fn object_id(&self) -> &str {
        &self.id
    }
}

fn safe_header_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|character| match character {
            '"' | '\r' | '\n' => '_',
            other => other,
        })
        .collect()
}

fn rfc3339_timestamp(timestamp: u64) -> String {
    DateTime::<Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
        .to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn openai_batch_job_key(batch: &OpenAiBatchRecord) -> String {
    format!("openai-batch:{}:{}", batch.tenant_id, batch.id)
}

fn anthropic_batch_job_key(batch: &AnthropicBatchRecord) -> String {
    format!("anthropic-batch:{}:{}", batch.tenant_id, batch.id)
}

fn batch_item_concurrency() -> usize {
    std::env::var("MIYA_BATCH_ITEM_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_BATCH_ITEM_CONCURRENCY)
}

fn accumulate_json_usage(usage: &mut ProviderUsage, value: &serde_json::Value) {
    let raw = value.get("usage").unwrap_or(&serde_json::Value::Null);
    let input = raw
        .get("input_tokens")
        .or_else(|| raw.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        .min(u64::from(u32::MAX)) as u32;
    let output = raw
        .get("output_tokens")
        .or_else(|| raw.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        .min(u64::from(u32::MAX)) as u32;
    usage.input_tokens = usage.input_tokens.saturating_add(input);
    usage.output_tokens = usage.output_tokens.saturating_add(output);
}
