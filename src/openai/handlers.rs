use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::OnceLock,
};

use axum::{
    Json as JsonExtractor,
    body::{Body, to_bytes},
    extract::{Extension, Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use parking_lot::RwLock;
use serde_json::{Value, json};

use crate::anthropic::{
    handlers::post_messages,
    middleware::{AppState, KeyContext},
};

use super::types::{
    AssistantParts, ChatCompletionRequest, OpenAIConversionError, OpenAIFunctionCall,
    OpenAIMessage, OpenAIToolCall, ResponsesRequest, ToolKindMap,
    assistant_message_for_history, assistant_parts_from_anthropic, chat_message_from_parts,
    chat_to_anthropic_with_metadata, finish_reason_from_anthropic, openai_error,
    response_output_from_parts_with_tools, response_tool_item, responses_to_chat_request,
    responses_usage_json, usage_json,
};

const MAX_COLLECT_BYTES: usize = 32 * 1024 * 1024;
const MAX_STORED_RESPONSES: usize = 512;

#[derive(Clone)]
struct StoredResponse {
    response: Value,
    messages: Vec<OpenAIMessage>,
}

static RESPONSES_STORE: OnceLock<RwLock<HashMap<String, StoredResponse>>> = OnceLock::new();

fn responses_store() -> &'static RwLock<HashMap<String, StoredResponse>> {
    RESPONSES_STORE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// POST /v1/chat/completions
pub async fn post_chat_completions(
    State(state): State<AppState>,
    Extension(key_ctx): Extension<KeyContext>,
    headers: HeaderMap,
    JsonExtractor(mut req): JsonExtractor<ChatCompletionRequest>,
) -> Response {
    apply_model_mapping(&state, &mut req.model);
    let metadata = resolve_session_metadata(req.prompt_cache_key.as_deref(), &headers);
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let converted = match chat_to_anthropic_with_metadata(&req, metadata) {
        Ok(converted) => converted,
        Err(e) => return conversion_error(e),
    };
    let stream = converted.anthropic.stream;
    let model = converted.anthropic.model.clone();

    let anthropic_response = post_messages(
        State(state),
        Extension(key_ctx),
        JsonExtractor(converted.anthropic),
    )
    .await;

    if stream {
        convert_chat_stream_response(anthropic_response, model, include_usage).await
    } else {
        convert_chat_non_stream_response(anthropic_response).await
    }
}

/// POST /v1/responses
pub async fn post_responses(
    State(state): State<AppState>,
    Extension(key_ctx): Extension<KeyContext>,
    headers: HeaderMap,
    JsonExtractor(mut req): JsonExtractor<ResponsesRequest>,
) -> Response {
    if let Some(model) = req.model.as_mut() {
        apply_model_mapping(&state, model);
    }
    let metadata = resolve_session_metadata(req.prompt_cache_key.as_deref(), &headers);
    let response_config = ResponsesResponseConfig::from_request(&req);
    let previous_messages = match load_previous_messages(req.previous_response_id.as_deref()) {
        Ok(messages) => messages,
        Err(resp) => return resp,
    };
    let chat_req = match responses_to_chat_request(&req, previous_messages) {
        Ok(req) => req,
        Err(e) => return conversion_error(e),
    };
    let converted = match chat_to_anthropic_with_metadata(&chat_req, metadata) {
        Ok(converted) => converted,
        Err(e) => return conversion_error(e),
    };

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = unix_ts();
    let store_response = req.store.unwrap_or(true);
    let stream = converted.anthropic.stream;
    let model = converted.anthropic.model.clone();
    let messages_for_history = converted.openai_messages.clone();
    let tool_kinds = converted.tool_kinds.clone();

    let anthropic_response = post_messages(
        State(state),
        Extension(key_ctx),
        JsonExtractor(converted.anthropic),
    )
    .await;

    if stream {
        convert_responses_stream_response(
            anthropic_response,
            ResponsesStreamMeta {
                response_id,
                created_at,
                model,
                previous_response_id: req.previous_response_id.clone(),
                metadata: req.metadata.clone(),
                store_response,
                messages_for_history,
                tool_kinds,
                response_config,
            },
        )
        .await
    } else {
        convert_responses_non_stream_response(
            anthropic_response,
            response_id,
            created_at,
            req.previous_response_id.clone(),
            req.metadata.clone(),
            store_response,
            messages_for_history,
            tool_kinds,
            response_config,
        )
        .await
    }
}

fn resolve_session_metadata(
    prompt_cache_key: Option<&str>,
    headers: &HeaderMap,
) -> Option<crate::anthropic::types::Metadata> {
    let candidates = [
        prompt_cache_key,
        headers
            .get("x-session-affinity")
            .and_then(|value| value.to_str().ok()),
        headers
            .get("x-client-request-id")
            .and_then(|value| value.to_str().ok()),
        headers.get("session_id").and_then(|value| value.to_str().ok()),
    ];
    candidates.into_iter().flatten().find_map(|candidate| {
        let raw_uuid = candidate.strip_prefix("session_").unwrap_or(candidate);
        let uuid = uuid::Uuid::parse_str(raw_uuid).ok()?;
        Some(crate::anthropic::types::Metadata {
            user_id: Some(format!("session_{uuid}")),
        })
    })
}

#[derive(Clone)]
struct ResponsesResponseConfig {
    parallel_tool_calls: bool,
    tool_choice: Value,
    tools: Vec<Value>,
}

impl ResponsesResponseConfig {
    fn from_request(req: &ResponsesRequest) -> Self {
        let mut tools = req
            .tools
            .iter()
            .filter_map(|tool| serde_json::to_value(tool).ok())
            .collect::<Vec<_>>();
        if let Some(Value::Array(items)) = &req.input {
            for item in items {
                if item.get("type").and_then(Value::as_str) == Some("additional_tools")
                    && let Some(extra) = item.get("tools").and_then(Value::as_array)
                {
                    tools.extend(extra.iter().cloned());
                }
            }
        }
        Self {
            parallel_tool_calls: req.parallel_tool_calls,
            tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
            tools,
        }
    }
}

/// GET /v1/responses/{id}
pub async fn get_response(Path(id): Path<String>) -> Response {
    if let Some(stored) = responses_store().read().get(&id).cloned() {
        return (StatusCode::OK, Json(stored.response)).into_response();
    }
    openai_status_error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        format!("response not found: {}", id),
    )
}

/// DELETE /v1/responses/{id}
pub async fn delete_response(Path(id): Path<String>) -> Response {
    let deleted = responses_store().write().remove(&id).is_some();
    (
        StatusCode::OK,
        Json(json!({
            "id": id,
            "object": "response.deleted",
            "deleted": deleted,
        })),
    )
        .into_response()
}

/// 请求时应用模型映射：命中配置的源模型名则原地改写为目标模型名。
///
/// 在 `chat_to_anthropic` 的启发式映射（gpt-*/o1/o3/codex → 默认兼容模型）之前执行，
/// 因此显式映射优先级更高；改写后的目标名（如 claude-opus-4.8）不匹配启发式前缀，
/// 会被透传，不会被二次改写。
fn apply_model_mapping(state: &AppState, model: &mut String) {
    if let Some(mappings) = &state.model_mappings
        && let Some(target) = mappings.resolve(model)
    {
        tracing::debug!("模型映射命中: {} → {}", model, target);
        *model = target;
    }
}

fn conversion_error(e: OpenAIConversionError) -> Response {
    openai_status_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.message)
}

fn openai_status_error(
    status: StatusCode,
    error_type: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    (status, Json(openai_error(message, error_type))).into_response()
}

fn load_previous_messages(previous_response_id: Option<&str>) -> Result<Vec<OpenAIMessage>, Response> {
    let Some(id) = previous_response_id else {
        return Ok(Vec::new());
    };
    responses_store()
        .read()
        .get(id)
        .map(|stored| stored.messages.clone())
        .ok_or_else(|| {
            openai_status_error(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("previous_response_id not found: {}", id),
            )
        })
}

async fn convert_chat_non_stream_response(response: Response) -> Response {
    let status = response.status();
    let body = response.into_body();
    if !status.is_success() {
        return convert_error_body(status, body).await;
    }

    let value = match collect_json_body(body).await {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    let parts = assistant_parts_from_anthropic(&value);
    (
        StatusCode::OK,
        Json(json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            "object": "chat.completion",
            "created": unix_ts(),
            "model": parts.model,
            "choices": [{
                "index": 0,
                "message": chat_message_from_parts(&parts),
                "logprobs": null,
                "finish_reason": finish_reason_from_anthropic(&parts.stop_reason),
            }],
            "usage": usage_json(&parts),
        })),
    )
        .into_response()
}

async fn convert_responses_non_stream_response(
    response: Response,
    response_id: String,
    created_at: i64,
    previous_response_id: Option<String>,
    metadata: Option<Value>,
    store_response: bool,
    mut messages_for_history: Vec<OpenAIMessage>,
    tool_kinds: ToolKindMap,
    response_config: ResponsesResponseConfig,
) -> Response {
    let status = response.status();
    let body = response.into_body();
    if !status.is_success() {
        return convert_error_body(status, body).await;
    }

    let value = match collect_json_body(body).await {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    let parts = assistant_parts_from_anthropic(&value);
    let response_obj = build_responses_object(
        &response_id,
        created_at,
        previous_response_id,
        metadata,
        &parts,
        &tool_kinds,
        &response_config,
        "completed",
    );

    messages_for_history.push(assistant_message_for_history(&parts));
    if store_response {
        save_response(response_id, response_obj.clone(), messages_for_history);
    }

    (StatusCode::OK, Json(response_obj)).into_response()
}

async fn collect_json_body(body: Body) -> Result<Value, Response> {
    let bytes = match to_bytes(body, MAX_COLLECT_BYTES).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return Err(openai_status_error(
                StatusCode::BAD_GATEWAY,
                "server_error",
                format!("failed to read upstream response: {}", e),
            ));
        }
    };
    serde_json::from_slice(&bytes).map_err(|e| {
        openai_status_error(
            StatusCode::BAD_GATEWAY,
            "server_error",
            format!("failed to parse upstream response: {}", e),
        )
    })
}

async fn convert_error_body(status: StatusCode, body: Body) -> Response {
    let bytes = to_bytes(body, MAX_COLLECT_BYTES).await.unwrap_or_default();
    let parsed = serde_json::from_slice::<Value>(&bytes).ok();
    let message = parsed
        .as_ref()
        .and_then(|v| v.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).to_string());
    let error_type = parsed
        .as_ref()
        .and_then(|v| v.pointer("/error/type"))
        .and_then(Value::as_str)
        .unwrap_or("server_error");
    openai_status_error(status, error_type, message)
}

async fn convert_chat_stream_response(
    response: Response,
    model: String,
    include_usage: bool,
) -> Response {
    let status = response.status();
    if !status.is_success() {
        return convert_error_body(status, response.into_body()).await;
    }

    let stream = transform_anthropic_sse(
        response.into_body(),
        ChatStreamTranslator::new(model, include_usage),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

struct ResponsesStreamMeta {
    response_id: String,
    created_at: i64,
    model: String,
    previous_response_id: Option<String>,
    metadata: Option<Value>,
    store_response: bool,
    messages_for_history: Vec<OpenAIMessage>,
    tool_kinds: ToolKindMap,
    response_config: ResponsesResponseConfig,
}

async fn convert_responses_stream_response(response: Response, meta: ResponsesStreamMeta) -> Response {
    let status = response.status();
    if !status.is_success() {
        return convert_error_body(status, response.into_body()).await;
    }

    let stream = transform_anthropic_sse(response.into_body(), ResponsesStreamTranslator::new(meta));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

trait AnthropicSseTranslator {
    fn handle_frame(&mut self, frame: SseFrame) -> Vec<Bytes>;
    fn finish(&mut self) -> Vec<Bytes>;
    fn stream_error(&mut self, message: String) -> Vec<Bytes>;
    fn is_terminal(&self) -> bool;
}

fn transform_anthropic_sse<T>(
    body: Body,
    translator: T,
) -> impl Stream<Item = Result<Bytes, Infallible>>
where
    T: AnthropicSseTranslator + Send + 'static,
{
    let data_stream = body.into_data_stream();
    stream::unfold(
        (data_stream, SseFrameParser::default(), translator, false),
        |(mut data_stream, mut parser, mut translator, mut finished)| async move {
            if finished {
                return None;
            }

            loop {
                match data_stream.next().await {
                    Some(Ok(chunk)) => {
                        let frames = parser.push(&chunk);
                        let mut out = Vec::new();
                        for frame in frames {
                            out.extend(translator.handle_frame(frame).into_iter().map(Ok));
                        }
                        if translator.is_terminal() {
                            finished = true;
                        }
                        if !out.is_empty() {
                            return Some((
                                stream::iter(out),
                                (data_stream, parser, translator, finished),
                            ));
                        }
                    }
                    Some(Err(e)) => {
                        finished = true;
                        let bytes = translator.stream_error(format!("upstream stream error: {e}"));
                        return Some((
                            stream::iter(bytes.into_iter().map(Ok).collect::<Vec<_>>()),
                            (data_stream, parser, translator, finished),
                        ));
                    }
                    None => {
                        finished = true;
                        let mut out = Vec::new();
                        for frame in parser.finish() {
                            out.extend(translator.handle_frame(frame).into_iter().map(Ok));
                        }
                        out.extend(translator.finish().into_iter().map(Ok));
                        if out.is_empty() {
                            return None;
                        }
                        return Some((
                            stream::iter(out),
                            (data_stream, parser, translator, finished),
                        ));
                    }
                }
            }
        },
    )
    .flatten()
}

#[derive(Default)]
struct SseFrameParser {
    buffer: String,
}

#[derive(Debug)]
struct SseFrame {
    event: String,
    data: String,
}

impl SseFrameParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<SseFrame> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut frames = Vec::new();
        while let Some(pos) = self.buffer.find("\n\n") {
            let raw = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();
            if let Some(frame) = parse_sse_frame(&raw) {
                frames.push(frame);
            }
        }
        frames
    }

    fn finish(&mut self) -> Vec<SseFrame> {
        let raw = std::mem::take(&mut self.buffer);
        parse_sse_frame(&raw).into_iter().collect()
    }
}

fn parse_sse_frame(raw: &str) -> Option<SseFrame> {
    let mut event = String::new();
    let mut data_lines = Vec::new();
    for line in raw.lines().map(|line| line.trim_end_matches('\r')) {
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
        }
    }
    if event.is_empty() && data_lines.is_empty() {
        return None;
    }
    Some(SseFrame {
        event,
        data: data_lines.join("\n"),
    })
}

struct ToolStreamAcc {
    id: String,
    name: String,
    args: String,
    index: usize,
}

struct ChatStreamTranslator {
    id: String,
    model: String,
    created: i64,
    include_usage: bool,
    sent_role: bool,
    done: bool,
    finish_reason: Option<String>,
    usage: Option<Value>,
    tools: HashMap<i64, ToolStreamAcc>,
    next_tool_index: usize,
}

impl ChatStreamTranslator {
    fn new(model: String, include_usage: bool) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            model,
            created: unix_ts(),
            include_usage,
            sent_role: false,
            done: false,
            finish_reason: None,
            usage: None,
            tools: HashMap::new(),
            next_tool_index: 0,
        }
    }

    fn ensure_role(&mut self) -> Vec<Bytes> {
        if self.sent_role {
            return Vec::new();
        }
        self.sent_role = true;
        vec![self.chunk(json!({"role": "assistant"}), None, None)]
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>, usage: Option<Value>) -> Bytes {
        chat_data_sse(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "logprobs": null,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }))
    }
}

impl AnthropicSseTranslator for ChatStreamTranslator {
    fn handle_frame(&mut self, frame: SseFrame) -> Vec<Bytes> {
        if self.done || frame.event == "ping" {
            return Vec::new();
        }
        let data = match serde_json::from_str::<Value>(&frame.data) {
            Ok(data) => data,
            Err(_) => return Vec::new(),
        };

        match frame.event.as_str() {
            "message_start" => self.ensure_role(),
            "content_block_start" => {
                if data
                    .pointer("/content_block/type")
                    .and_then(Value::as_str)
                    == Some("tool_use")
                {
                    let block_index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                    self.tools.insert(
                        block_index,
                        ToolStreamAcc {
                            id: data
                                .pointer("/content_block/id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: data
                                .pointer("/content_block/name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            args: String::new(),
                            index: self.next_tool_index,
                        },
                    );
                    self.next_tool_index += 1;
                }
                Vec::new()
            }
            "content_block_delta" => {
                let mut out = self.ensure_role();
                match data.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = data.pointer("/delta/text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            out.push(self.chunk(json!({"content": text}), None, None));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = data.pointer("/delta/thinking").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            out.push(self.chunk(json!({"reasoning_content": text}), None, None));
                        }
                    }
                    Some("input_json_delta") => {
                        let index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                        if let Some(tool) = self.tools.get_mut(&index)
                            && let Some(delta) =
                                data.pointer("/delta/partial_json").and_then(Value::as_str)
                        {
                            tool.args.push_str(delta);
                        }
                    }
                    _ => {}
                }
                out
            }
            "content_block_stop" => {
                let index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                let Some(tool) = self.tools.remove(&index) else {
                    return Vec::new();
                };
                let mut out = self.ensure_role();
                out.push(self.chunk(
                    json!({
                        "tool_calls": [{
                            "index": tool.index,
                            "id": tool.id,
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "arguments": tool.args,
                            }
                        }]
                    }),
                    None,
                    None,
                ));
                out
            }
            "message_delta" => {
                self.finish_reason = data
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(finish_reason_from_anthropic)
                    .map(str::to_string);
                self.usage = Some(usage_from_anthropic_delta(&data));
                Vec::new()
            }
            "message_stop" => self.finish(),
            "error" => {
                self.done = true;
                vec![
                    chat_data_sse(json!({
                        "error": data.get("error").cloned().unwrap_or(data)
                    })),
                    Bytes::from_static(b"data: [DONE]\n\n"),
                ]
            }
            _ => Vec::new(),
        }
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        let mut out = self.ensure_role();
        let finish_reason = self.finish_reason.as_deref().unwrap_or("stop");
        let usage = self.include_usage.then(|| self.usage.clone().unwrap_or_else(|| json!(null)));
        out.push(self.chunk(json!({}), Some(finish_reason), usage));
        out.push(Bytes::from_static(b"data: [DONE]\n\n"));
        out
    }

    fn stream_error(&mut self, message: String) -> Vec<Bytes> {
        self.done = true;
        vec![
            chat_data_sse(json!({
                "error": { "message": message, "type": "server_error" }
            })),
            Bytes::from_static(b"data: [DONE]\n\n"),
        ]
    }

    fn is_terminal(&self) -> bool {
        self.done
    }
}

struct ResponsesStreamTranslator {
    meta: ResponsesStreamMeta,
    done: bool,
    created_sent: bool,
    message_started: bool,
    message_done: bool,
    message_item_id: String,
    output_index: usize,
    text: String,
    reasoning: String,
    tool_calls: Vec<OpenAIToolCall>,
    tools: HashMap<i64, ToolStreamAcc>,
    usage: Option<Value>,
    stop_reason: Option<String>,
    saw_message_stop: bool,
    sequence_number: Cell<i64>,
    reasoning_started: bool,
    reasoning_done: bool,
    reasoning_item_id: String,
    reasoning_output_index: Option<usize>,
    reasoning_blocks: HashSet<i64>,
    web_searches: HashMap<i64, WebSearchStreamAcc>,
    completed_web_searches: Vec<(String, String)>,
}

struct WebSearchStreamAcc {
    id: String,
    query: String,
    output_index: usize,
}

impl ResponsesStreamTranslator {
    fn new(meta: ResponsesStreamMeta) -> Self {
        Self {
            meta,
            done: false,
            created_sent: false,
            message_started: false,
            message_done: false,
            message_item_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            output_index: 0,
            text: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            tools: HashMap::new(),
            usage: None,
            stop_reason: None,
            saw_message_stop: false,
            sequence_number: Cell::new(0),
            reasoning_started: false,
            reasoning_done: false,
            reasoning_item_id: format!("rs_{}", uuid::Uuid::new_v4().simple()),
            reasoning_output_index: None,
            reasoning_blocks: HashSet::new(),
            web_searches: HashMap::new(),
            completed_web_searches: Vec::new(),
        }
    }

    fn emit(&self, event: &str, mut payload: Value) -> Bytes {
        payload["type"] = json!(event);
        let sequence_number = self.sequence_number.get();
        payload["sequence_number"] = json!(sequence_number);
        self.sequence_number.set(sequence_number + 1);
        responses_event_sse(event, payload)
    }

    fn created_response(&self, status: &str, output: Vec<Value>, output_text: &str) -> Value {
        let mut response = json!({
            "id": self.meta.response_id,
            "object": "response",
            "created_at": self.meta.created_at,
            "status": status,
            "model": self.meta.model,
            "previous_response_id": self.meta.previous_response_id,
            "output": output,
            "output_text": output_text,
            "parallel_tool_calls": self.meta.response_config.parallel_tool_calls,
            "tool_choice": self.meta.response_config.tool_choice,
            "tools": self.meta.response_config.tools,
        });
        if let Some(metadata) = &self.meta.metadata {
            response["metadata"] = metadata.clone();
        }
        response
    }

    fn ensure_created(&mut self) -> Vec<Bytes> {
        if self.created_sent {
            return Vec::new();
        }
        self.created_sent = true;
        // 同时补发 response.in_progress：官方 Responses 流与 Kiro-Go 都在 created
        // 之后紧跟一条 in_progress，部分 OpenAI SDK 以此判定流已正常开始。
        vec![
            self.emit(
                "response.created",
                json!({
                    "response": self.created_response("in_progress", Vec::new(), ""),
                }),
            ),
            self.emit(
                "response.in_progress",
                json!({
                    "response": self.created_response("in_progress", Vec::new(), ""),
                }),
            ),
        ]
    }

    fn ensure_message_started(&mut self) -> Vec<Bytes> {
        let mut out = self.ensure_created();
        if self.message_started {
            return out;
        }
        self.message_started = true;
        out.push(self.emit(
            "response.output_item.added",
            json!({
                "output_index": self.output_index,
                "item": {
                    "id": self.message_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            }),
        ));
        out.push(self.emit(
            "response.content_part.added",
            json!({
                "item_id": self.message_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                }
            }),
        ));
        out
    }

    fn close_message(&mut self) -> Vec<Bytes> {
        if !self.message_started || self.message_done {
            return Vec::new();
        }
        self.message_done = true;
        let item = json!({
            "id": self.message_item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": self.text,
                "annotations": [],
            }],
        });
        self.output_index += 1;
        vec![
            self.emit(
                "response.output_text.done",
                json!({
                    "item_id": self.message_item_id,
                    "output_index": self.output_index - 1,
                    "content_index": 0,
                    "text": self.text,
                }),
            ),
            self.emit(
                "response.content_part.done",
                json!({
                    "item_id": self.message_item_id,
                    "output_index": self.output_index - 1,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": self.text,
                        "annotations": [],
                    }
                }),
            ),
            self.emit(
                "response.output_item.done",
                json!({
                    "output_index": self.output_index - 1,
                    "item": item,
                }),
            ),
        ]
    }

    fn emit_reasoning_delta(&mut self, delta: &str) -> Vec<Bytes> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if !self.reasoning_started {
            self.reasoning_started = true;
            self.reasoning_output_index = Some(self.output_index);
            self.output_index += 1;
            out.push(self.emit(
                "response.output_item.added",
                json!({
                    "output_index": self.reasoning_output_index,
                    "item": {
                        "id": self.reasoning_item_id,
                        "type": "reasoning",
                        "summary": [],
                    }
                }),
            ));
        }
        self.reasoning.push_str(delta);
        out.push(self.emit(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": self.reasoning_output_index,
                "summary_index": 0,
                "delta": delta,
            }),
        ));
        out
    }

    fn close_reasoning(&mut self) -> Vec<Bytes> {
        if !self.reasoning_started || self.reasoning_done {
            return Vec::new();
        }
        self.reasoning_done = true;
        let item = json!({
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": self.reasoning }],
            "content": null,
        });
        vec![
            self.emit(
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": self.reasoning_item_id,
                    "output_index": self.reasoning_output_index,
                    "summary_index": 0,
                    "text": self.reasoning,
                }),
            ),
            self.emit(
                "response.output_item.done",
                json!({ "output_index": self.reasoning_output_index, "item": item }),
            ),
        ]
    }
}

impl AnthropicSseTranslator for ResponsesStreamTranslator {
    fn handle_frame(&mut self, frame: SseFrame) -> Vec<Bytes> {
        if self.done || frame.event == "ping" {
            return Vec::new();
        }
        let data = match serde_json::from_str::<Value>(&frame.data) {
            Ok(data) => data,
            Err(_) => return Vec::new(),
        };

        let mut prefixed = Vec::new();
        if let Some(thinking) = data
            .get("kiro_thinking")
            .and_then(Value::as_str)
            .filter(|thinking| !thinking.is_empty())
        {
            prefixed.extend(self.ensure_created());
            prefixed.extend(self.emit_reasoning_delta(thinking));
            prefixed.extend(self.close_reasoning());
        }

        let translated = match frame.event.as_str() {
            "message_start" => {
                if let Some(usage) = data.pointer("/message/usage") {
                    self.usage = Some(merge_usage(self.usage.take(), usage));
                }
                self.ensure_created()
            }
            "content_block_start" => {
                let block_index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                let block_type = data
                    .pointer("/content_block/type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match block_type {
                    "tool_use" => {
                        self.tools.insert(
                            block_index,
                            ToolStreamAcc {
                                id: data
                                    .pointer("/content_block/id")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                name: data
                                    .pointer("/content_block/name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                args: String::new(),
                                index: self.output_index,
                            },
                        );
                        Vec::new()
                    }
                    "thinking" => {
                        self.reasoning_blocks.insert(block_index);
                        Vec::new()
                    }
                    "server_tool_use"
                        if data.pointer("/content_block/name").and_then(Value::as_str)
                            == Some("web_search") =>
                    {
                        let id = data
                            .pointer("/content_block/id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let query = data
                            .pointer("/content_block/input/query")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let output_index = self.output_index;
                        self.output_index += 1;
                        self.web_searches.insert(
                            block_index,
                            WebSearchStreamAcc {
                                id: id.clone(),
                                query: query.clone(),
                                output_index,
                            },
                        );
                        vec![self.emit(
                            "response.output_item.added",
                            json!({
                                "output_index": output_index,
                                "item": {
                                    "id": id,
                                    "type": "web_search_call",
                                    "status": "in_progress",
                                    "action": { "type": "search", "query": query },
                                }
                            }),
                        )]
                    }
                    _ => Vec::new(),
                }
            }
            "content_block_delta" => match data.pointer("/delta/type").and_then(Value::as_str) {
                Some("text_delta") => {
                    let mut out = self.ensure_message_started();
                    if let Some(delta) = data.pointer("/delta/text").and_then(Value::as_str)
                        && !delta.is_empty()
                    {
                        self.text.push_str(delta);
                        out.push(self.emit(
                            "response.output_text.delta",
                            json!({
                                "item_id": self.message_item_id,
                                "output_index": self.output_index,
                                "content_index": 0,
                                "delta": delta,
                            }),
                        ));
                    }
                    out
                }
                Some("thinking_delta") => {
                    let delta = data
                        .pointer("/delta/thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    self.emit_reasoning_delta(delta)
                }
                Some("input_json_delta") => {
                    let index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                    if let Some(tool) = self.tools.get_mut(&index)
                        && let Some(delta) =
                            data.pointer("/delta/partial_json").and_then(Value::as_str)
                    {
                        tool.args.push_str(delta);
                    }
                    Vec::new()
                }
                _ => Vec::new(),
            },
            "content_block_stop" => {
                let index = data.get("index").and_then(Value::as_i64).unwrap_or(0);
                if self.reasoning_blocks.remove(&index) {
                    prefixed.extend(self.close_reasoning());
                    return prefixed;
                }
                if let Some(search) = self.web_searches.remove(&index) {
                    let id = search.id;
                    let query = search.query;
                    let item = json!({
                        "id": id,
                        "type": "web_search_call",
                        "status": "completed",
                        "action": { "type": "search", "query": query },
                    });
                    self.completed_web_searches
                        .push((id.clone(), query.clone()));
                    prefixed.push(self.emit(
                        "response.output_item.done",
                        json!({ "output_index": search.output_index, "item": item }),
                    ));
                    return prefixed;
                }
                let Some(tool) = self.tools.remove(&index) else {
                    return Vec::new();
                };
                let mut out = self.close_message();
                let call = OpenAIToolCall {
                    id: Some(tool.id.clone()),
                    tool_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: tool.name.clone(),
                        arguments: tool.args.clone(),
                    },
                    namespace: None,
                };
                let item = response_tool_item(&call, &self.meta.tool_kinds, "completed");
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or("function_call");
                let item_id = item.get("id").cloned().unwrap_or(Value::Null);
                let added_item = match item_type {
                    "custom_tool_call" => {
                        let mut added = item.clone();
                        added["status"] = json!("in_progress");
                        added
                    }
                    _ => {
                        let mut added = item.clone();
                        added["status"] = json!("in_progress");
                        added["arguments"] = json!("");
                        added
                    }
                };
                self.tool_calls.push(call);
                out.push(
                    self.emit(
                        "response.output_item.added",
                        json!({
                            "output_index": self.output_index,
                            "item": added_item,
                        }),
                    )
                );
                if item_type == "custom_tool_call" {
                    let input = item.get("input").cloned().unwrap_or(json!(""));
                    out.extend([
                        self.emit(
                            "response.custom_tool_call_input.delta",
                            json!({
                                "item_id": item_id,
                                "output_index": self.output_index,
                                "delta": input,
                            }),
                        ),
                        self.emit(
                            "response.custom_tool_call_input.done",
                            json!({
                                "item_id": item_id,
                                "output_index": self.output_index,
                                "input": input,
                            }),
                        ),
                    ]);
                } else {
                    out.extend([
                    self.emit(
                        "response.function_call_arguments.delta",
                        json!({
                            "item_id": item_id,
                            "output_index": self.output_index,
                            "delta": item["arguments"],
                        }),
                    ),
                    self.emit(
                        "response.function_call_arguments.done",
                        json!({
                            "item_id": item_id,
                            "output_index": self.output_index,
                            "arguments": item["arguments"],
                        }),
                    ),
                    ]);
                }
                out.push(
                    self.emit(
                        "response.output_item.done",
                        json!({
                            "output_index": self.output_index,
                            "item": item,
                        }),
                    )
                );
                self.output_index += 1;
                out
            }
            "message_delta" => {
                self.stop_reason = data
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                // 保存 Anthropic 原始 usage（未汇总的 input/cache 分量），
                // 供 finish() 组装 Responses 口径时避免与 cache 重复相加。
                if let Some(usage) = data.get("usage") {
                    self.usage = Some(merge_usage(self.usage.take(), usage));
                }
                Vec::new()
            }
            "message_stop" => {
                self.saw_message_stop = true;
                self.finish()
            }
            "error" => {
                self.done = true;
                let mut out = self.ensure_created();
                out.push(self.emit(
                    "response.failed",
                    json!({
                        "response": {
                            "id": self.meta.response_id,
                            "object": "response",
                            "created_at": self.meta.created_at,
                            "status": "failed",
                            "model": self.meta.model,
                            "error": data.get("error").cloned().unwrap_or(data),
                        }
                    }),
                ));
                out.push(Bytes::from_static(b"data: [DONE]\n\n"));
                out
            }
            _ => Vec::new(),
        };
        prefixed.extend(translated);
        prefixed
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.done {
            return Vec::new();
        }
        if !self.saw_message_stop {
            return self.stream_error("upstream stream ended before message_stop".to_string());
        }
        self.done = true;
        let mut out = self.ensure_created();
        out.extend(self.close_reasoning());
        out.extend(self.close_message());

        let parts = AssistantParts {
            text: self.text.clone(),
            reasoning: self.reasoning.clone(),
            tool_calls: self.tool_calls.clone(),
            stop_reason: self.stop_reason.clone().unwrap_or_else(|| "end_turn".to_string()),
            input_tokens: self
                .usage
                .as_ref()
                .and_then(|u| u.get("input_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            output_tokens: self
                .usage
                .as_ref()
                .and_then(|u| u.get("output_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            cache_read_tokens: self
                .usage
                .as_ref()
                .and_then(|u| u.get("cache_read_input_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            cache_creation_tokens: self
                .usage
                .as_ref()
                .and_then(|u| u.get("cache_creation_input_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            model: self.meta.model.clone(),
            web_searches: self.completed_web_searches.clone(),
            reasoning_tokens: self
                .usage
                .as_ref()
                .and_then(|usage| usage.get("reasoning_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            credit_usage: self
                .usage
                .as_ref()
                .and_then(|usage| usage.get("credit_usage"))
                .and_then(Value::as_f64),
            credit_unit: self
                .usage
                .as_ref()
                .and_then(|usage| usage.get("credit_unit"))
                .and_then(Value::as_str)
                .map(str::to_string),
            credit_unit_plural: self
                .usage
                .as_ref()
                .and_then(|usage| usage.get("credit_unit_plural"))
                .and_then(Value::as_str)
                .map(str::to_string),
        };
        let incomplete = matches!(
            self.stop_reason.as_deref(),
            Some("max_tokens" | "model_context_window_exceeded")
        );
        let status = if incomplete { "incomplete" } else { "completed" };
        let response = build_responses_object(
            &self.meta.response_id,
            self.meta.created_at,
            self.meta.previous_response_id.clone(),
            self.meta.metadata.clone(),
            &parts,
            &self.meta.tool_kinds,
            &self.meta.response_config,
            status,
        );

        if self.meta.store_response {
            let mut history = self.meta.messages_for_history.clone();
            history.push(assistant_message_for_history(&parts));
            save_response(self.meta.response_id.clone(), response.clone(), history);
        }

        let event = if incomplete { "response.incomplete" } else { "response.completed" };
        out.push(self.emit(
            event,
            json!({
                "response": response,
            }),
        ));
        out.push(Bytes::from_static(b"data: [DONE]\n\n"));
        out
    }

    fn stream_error(&mut self, message: String) -> Vec<Bytes> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        let mut out = self.ensure_created();
        let mut response = self.created_response("failed", Vec::new(), "");
        response["error"] = json!({ "code": "server_error", "message": message });
        out.push(self.emit(
            "response.failed",
            json!({ "response": response }),
        ));
        out.push(Bytes::from_static(b"data: [DONE]\n\n"));
        out
    }

    fn is_terminal(&self) -> bool {
        self.done
    }
}

fn usage_from_anthropic_delta(data: &Value) -> Value {
    let usage = data.get("usage").unwrap_or(&Value::Null);
    let input = usage
        .get("input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let cache_read = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let prompt = input + cache_creation + cache_read;
    let completion = usage
        .get("output_tokens")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
        }
    })
}

fn merge_usage(current: Option<Value>, incoming: &Value) -> Value {
    let mut merged = current.unwrap_or_else(|| json!({}));
    let Some(target) = merged.as_object_mut() else {
        return incoming.clone();
    };
    let Some(source) = incoming.as_object() else {
        return merged;
    };
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
    merged
}

fn build_responses_object(
    id: &str,
    created_at: i64,
    previous_response_id: Option<String>,
    metadata: Option<Value>,
    parts: &AssistantParts,
    tool_kinds: &ToolKindMap,
    response_config: &ResponsesResponseConfig,
    status: &str,
) -> Value {
    let (output, output_text) = response_output_from_parts_with_tools(parts, tool_kinds);
    let mut response = json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": parts.model,
        "previous_response_id": previous_response_id,
        "output": output,
        "output_text": output_text,
        "usage": responses_usage_json(parts),
        "parallel_tool_calls": response_config.parallel_tool_calls,
        "tool_choice": response_config.tool_choice,
        "tools": response_config.tools,
    });
    if let Some(metadata) = metadata {
        response["metadata"] = metadata;
    }
    if status == "incomplete" {
        response["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    }
    response
}

fn save_response(id: String, response: Value, messages: Vec<OpenAIMessage>) {
    let mut store = responses_store().write();
    if store.len() >= MAX_STORED_RESPONSES
        && let Some(first_key) = store.keys().next().cloned()
    {
        store.remove(&first_key);
    }
    store.insert(id, StoredResponse { response, messages });
}

fn chat_data_sse(value: Value) -> Bytes {
    Bytes::from(format!(
        "data: {}\n\n",
        serde_json::to_string(&value).unwrap_or_default()
    ))
}

fn responses_event_sse(event: &str, value: Value) -> Bytes {
    Bytes::from(format!(
        "event: {}\ndata: {}\n\n",
        event,
        serde_json::to_string(&value).unwrap_or_default()
    ))
}

fn unix_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use serde_json::{Value, json};

    use super::super::types::{
        ChatCompletionRequest, DeclaredTool, DeclaredToolKind, OpenAIMessage, ResponsesRequest,
        ToolKindMap, assistant_parts_from_anthropic, chat_to_anthropic,
        responses_input_to_messages, responses_to_chat_request,
    };
    use super::{
        AnthropicSseTranslator, ChatStreamTranslator, HeaderMap, ResponsesResponseConfig,
        ResponsesStreamMeta, ResponsesStreamTranslator, SseFrameParser, resolve_session_metadata,
    };

    /// 把一段 Anthropic SSE 原文喂给 ChatStreamTranslator，收集其产出的
    /// `data: {...}` 行并解析成 JSON 序列（chat completions 流不带 `event:` 行）。
    /// `data: [DONE]` 作为 Value::Null 占位返回，便于断言收尾。
    fn run_chat_translator(anthropic_sse: &str, include_usage: bool) -> Vec<Value> {
        let mut translator = ChatStreamTranslator::new("gpt-4o".to_string(), include_usage);
        let mut parser = SseFrameParser::default();
        let mut raw_out: Vec<Bytes> = Vec::new();
        for frame in parser.push(anthropic_sse.as_bytes()) {
            raw_out.extend(translator.handle_frame(frame));
        }
        for frame in parser.finish() {
            raw_out.extend(translator.handle_frame(frame));
        }
        raw_out.extend(translator.finish());

        let joined = raw_out
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect::<String>();
        let mut chunks = Vec::new();
        for block in joined.split("\n\n") {
            let Some(rest) = block.trim().strip_prefix("data:") else {
                continue;
            };
            let data = rest.trim();
            if data == "[DONE]" {
                chunks.push(Value::Null);
            } else if let Ok(v) = serde_json::from_str::<Value>(data) {
                chunks.push(v);
            }
        }
        chunks
    }

    /// 把一段 Anthropic SSE 原文喂给 translator，收集其产出的所有 SSE 事件帧，
    /// 解析成 (event_name, data_json) 序列。复刻 transform_anthropic_sse 的分帧逻辑，
    /// 但同步执行，方便断言。`data: [DONE]` 单独作为 ("", DONE 标记) 返回。
    fn run_responses_translator(anthropic_sse: &str) -> Vec<(String, Value)> {
        let meta = ResponsesStreamMeta {
            response_id: "resp_test".to_string(),
            created_at: 42,
            model: "claude-sonnet-4.5".to_string(),
            previous_response_id: None,
            metadata: None,
            store_response: false,
            messages_for_history: Vec::new(),
            tool_kinds: ToolKindMap::new(),
            response_config: ResponsesResponseConfig::from_request(
                &serde_json::from_value(json!({
                    "model": "claude-sonnet-4.5",
                    "input": "test"
                }))
                .unwrap(),
            ),
        };
        let mut translator = ResponsesStreamTranslator::new(meta);
        let mut parser = SseFrameParser::default();
        let mut raw_out: Vec<Bytes> = Vec::new();
        for frame in parser.push(anthropic_sse.as_bytes()) {
            raw_out.extend(translator.handle_frame(frame));
        }
        for frame in parser.finish() {
            raw_out.extend(translator.handle_frame(frame));
        }
        raw_out.extend(translator.finish());

        // 把产出的字节流重新按 SSE 帧解析
        let joined = raw_out
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect::<String>();
        let mut events = Vec::new();
        for block in joined.split("\n\n") {
            if block.trim().is_empty() {
                continue;
            }
            let mut event_name = String::new();
            let mut data = String::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    event_name = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data = rest.trim_start().to_string();
                }
            }
            if data == "[DONE]" {
                events.push(("[DONE]".to_string(), Value::Null));
            } else if let Ok(v) = serde_json::from_str::<Value>(&data) {
                events.push((event_name, v));
            }
        }
        events
    }

    #[test]
    fn gpt_5_6_responses_model_is_not_downgraded() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "input": "hello",
            "reasoning": { "effort": "high" }
        }))
        .unwrap();
        let chat = responses_to_chat_request(&req, Vec::new()).unwrap();
        let converted = chat_to_anthropic(&chat).unwrap();
        assert_eq!(converted.anthropic.model, "gpt-5.6-sol");
        assert_eq!(converted.anthropic.output_config.unwrap().effort, "high");
    }

    #[test]
    fn gpt_5_6_preserves_none_and_max_effort() {
        for effort in ["none", "max"] {
            let req: ResponsesRequest = serde_json::from_value(json!({
                "model": "gpt-5.6-terra",
                "input": "hello",
                "reasoning": { "effort": effort }
            }))
            .unwrap();
            let chat = responses_to_chat_request(&req, Vec::new()).unwrap();
            let converted = chat_to_anthropic(&chat).unwrap();
            assert_eq!(
                converted.anthropic.output_config.unwrap().effort,
                effort
            );
        }
    }

    #[test]
    fn responses_custom_and_namespace_tools_round_trip() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {"type": "custom", "name": "apply_patch"},
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}]
                        }
                    ]
                },
                {"type": "message", "role": "user", "content": "go"}
            ]
        }))
        .unwrap();
        let chat = responses_to_chat_request(&req, Vec::new()).unwrap();
        let converted = chat_to_anthropic(&chat).unwrap();
        let tools = converted.anthropic.tools.unwrap();
        assert!(tools.iter().any(|tool| tool.name == "apply_patch"));
        assert!(tools.iter().any(|tool| tool.name == "collaboration__spawn_agent"));
        assert_eq!(
            converted.tool_kinds["apply_patch"].kind,
            DeclaredToolKind::Custom
        );
        assert_eq!(
            converted.tool_kinds["collaboration__spawn_agent"]
                .namespace
                .as_deref(),
            Some("collaboration")
        );
    }

    #[test]
    fn responses_custom_stream_emits_matching_item_type() {
        let mut tool_kinds = ToolKindMap::new();
        tool_kinds.insert(
            "apply_patch".to_string(),
            DeclaredTool {
                kind: DeclaredToolKind::Custom,
                name: "apply_patch".to_string(),
                namespace: None,
            },
        );
        let meta = ResponsesStreamMeta {
            response_id: "resp_custom".to_string(),
            created_at: 1,
            model: "gpt-5.6-sol".to_string(),
            previous_response_id: None,
            metadata: None,
            store_response: false,
            messages_for_history: Vec::new(),
            tool_kinds,
            response_config: ResponsesResponseConfig::from_request(
                &serde_json::from_value(json!({"model": "gpt-5.6-sol", "input": "go"}))
                    .unwrap(),
            ),
        };
        let mut translator = ResponsesStreamTranslator::new(meta);
        let upstream = concat!(
            "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"apply_patch\"}}\n\n",
            "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"input\\\":\\\"PATCH\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"index\":0}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        let mut parser = SseFrameParser::default();
        let mut output = Vec::new();
        for frame in parser.push(upstream.as_bytes()) {
            output.extend(translator.handle_frame(frame));
        }
        let output = output
            .iter()
            .map(|bytes| String::from_utf8_lossy(bytes))
            .collect::<String>();
        assert!(output.contains("custom_tool_call"));
        assert!(output.contains("response.custom_tool_call_input.done"));
        assert!(output.contains("PATCH"));
        assert!(!output.contains("function_call_arguments.delta"));
    }

    #[test]
    fn responses_early_eof_is_failed_not_completed() {
        let meta = ResponsesStreamMeta {
            response_id: "resp_eof".to_string(),
            created_at: 1,
            model: "gpt-5.6-sol".to_string(),
            previous_response_id: None,
            metadata: None,
            store_response: false,
            messages_for_history: Vec::new(),
            tool_kinds: ToolKindMap::new(),
            response_config: ResponsesResponseConfig::from_request(
                &serde_json::from_value(json!({"model": "gpt-5.6-sol", "input": "go"}))
                    .unwrap(),
            ),
        };
        let mut translator = ResponsesStreamTranslator::new(meta);
        let output = translator
            .finish()
            .iter()
            .map(|bytes| String::from_utf8_lossy(bytes))
            .collect::<String>();
        assert!(output.contains("response.failed"));
        assert!(!output.contains("response.completed"));
    }

    #[test]
    fn session_metadata_prefers_prompt_cache_key() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-session-affinity",
            "67e55044-10b1-426f-9247-bb680e5fe0c8".parse().unwrap(),
        );
        let metadata = resolve_session_metadata(
            Some("550e8400-e29b-41d4-a716-446655440000"),
            &headers,
        )
        .unwrap();
        assert_eq!(
            metadata.user_id.as_deref(),
            Some("session_550e8400-e29b-41d4-a716-446655440000")
        );
    }

    #[test]
    fn chat_request_converts_tools_and_tool_results() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "user", "content": "summarize"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }]
        }))
        .unwrap();

        let converted = chat_to_anthropic(&req).unwrap();
        assert_eq!(converted.anthropic.model, "claude-sonnet-4.5");
        assert_eq!(converted.anthropic.messages.len(), 4);
        assert_eq!(converted.anthropic.system.unwrap()[0].text, "be brief");
        assert_eq!(converted.anthropic.tools.unwrap()[0].name, "get_weather");
    }

    /// OpenAI/Codex 的内置 web search 工具（type=web_search / web_search_preview）
    /// 必须转成 Anthropic 原生格式（name=web_search, type 以 web_search_ 开头），
    /// 否则会在 convert_openai_tools 的 `!= "function"` 分支被丢弃，联网搜索彻底失效。
    /// 后端 is_native_web_search_tool 要求 type.starts_with("web_search_")，
    /// 注意裸 "web_search" 不满足该前缀，必须归一化。
    #[test]
    fn chat_request_converts_web_search_tool_variants() {
        for typ in ["web_search", "web_search_preview", "web_search_preview_2025_03_11"] {
            let req: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "查一下今天的新闻"}],
                "tools": [{"type": typ}]
            }))
            .unwrap();
            let converted = chat_to_anthropic(&req).unwrap();
            let tools = converted
                .anthropic
                .tools
                .unwrap_or_else(|| panic!("web search 工具被丢弃了: type={typ}"));
            assert_eq!(tools.len(), 1, "type={typ}");
            assert_eq!(tools[0].name, "web_search", "type={typ}");
            let tt = tools[0].tool_type.as_deref().unwrap_or("");
            assert!(
                tt.starts_with("web_search_"),
                "type={typ} 归一化后 tool_type={tt} 不满足后端 starts_with(web_search_)"
            );
            assert_eq!(tools[0].max_uses, Some(8), "type={typ}");
        }
    }

    /// web search 与普通 function 工具混用时（Codex 常见场景），两者都要保留：
    /// web_search 转原生、function 照常转，交给后端 agentic loop。
    #[test]
    fn chat_request_keeps_web_search_alongside_function_tools() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "查天气并联网核对"}],
            "tools": [
                {"type": "web_search"},
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}
            ]
        }))
        .unwrap();
        let converted = chat_to_anthropic(&req).unwrap();
        let tools = converted.anthropic.tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t.name == "web_search"
            && t.tool_type.as_deref().is_some_and(|s| s.starts_with("web_search_"))));
        assert!(tools.iter().any(|t| t.name == "get_weather" && t.tool_type.is_none()));
    }

    /// 回归：OpenAI 并行工具调用（一条 assistant 带多个 tool_calls + 多条独立 tool
    /// 消息）必须合并成 assistant[tool_use...] + 单条 user[tool_result...]，否则连续
    /// user 消息会破坏配对，上游报 400 "tool_use and tool_result blocks must be
    /// correctly paired and ordered"。
    #[test]
    fn chat_request_batches_parallel_tool_results_into_one_user_message() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role": "user", "content": "查北京和上海的天气"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"上海\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_a", "content": "北京晴"},
                {"role": "tool", "tool_call_id": "call_b", "content": "上海雨"},
                {"role": "user", "content": "总结"}
            ]
        }))
        .unwrap();

        let converted = chat_to_anthropic(&req).unwrap();
        let msgs = &converted.anthropic.messages;
        // user / assistant(2×tool_use) / user(2×tool_result 合并) / user(总结)
        assert_eq!(msgs.len(), 4);

        // assistant 轮次带两个 tool_use
        assert_eq!(msgs[1].role, "assistant");
        let assistant_blocks = msgs[1].content.as_array().unwrap();
        let tool_uses = assistant_blocks.iter().filter(|b| b["type"] == "tool_use").count();
        assert_eq!(tool_uses, 2);

        // 两个 tool_result 必须在同一条 user 消息里（关键：不能拆成两条）
        assert_eq!(msgs[2].role, "user");
        let results = msgs[2].content.as_array().unwrap();
        assert_eq!(results.len(), 2, "两个 tool_result 必须合并进一条 user 消息");
        assert!(results.iter().all(|r| r["type"] == "tool_result"));
        assert_eq!(results[0]["tool_use_id"], "call_a");
        assert_eq!(results[1]["tool_use_id"], "call_b");

        // 不应出现连续两条 user 消息承载 tool_result
        assert_eq!(msgs[3].role, "user"); // 这是"总结"，与上一条 tool_result user 相邻是允许的
    }

    #[test]
    fn responses_input_merges_parallel_function_calls() {
        let messages = responses_input_to_messages(Some(&json!([
            {"type": "function_call", "call_id": "call_a", "name": "a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_b", "name": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "ok"}
        ])))
        .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[0].tool_calls.len(), 2);
        assert_eq!(messages[1].role, "tool");
    }

    /// Codex 通过 responses 请求体 `reasoning:{effort}` 下发档位（none/minimal/low/
    /// medium/high/xhigh）。验证：
    /// - none → 不下发 thinking（关推理），否则后端 parse("none") 失败会 fallback 成 high；
    /// - minimal → 归一化到后端认得的 low（后端 EffortTier 无 minimal）；
    /// - xhigh 透传，high/medium 原样；
    /// - 归一化后的 effort 必须是后端 EffortTier 认得的值。
    #[test]
    fn responses_reasoning_effort_is_recognized_and_normalized() {
        let build = |effort: &str| -> ResponsesRequest {
            serde_json::from_value(json!({
                "model": "gpt-5.5",
                "input": "hi",
                "reasoning": { "effort": effort }
            }))
            .unwrap()
        };

        // none → 关推理：无 thinking、无 output_config
        let chat = responses_to_chat_request(&build("none"), Vec::new()).unwrap();
        let anthropic = chat_to_anthropic(&chat).unwrap().anthropic;
        assert!(anthropic.thinking.is_none(), "none 不应开启 thinking");
        assert!(anthropic.output_config.is_none(), "none 不应下发 output_config");

        // minimal → 归一化到 low（后端无 minimal 档）
        let chat = responses_to_chat_request(&build("minimal"), Vec::new()).unwrap();
        let anthropic = chat_to_anthropic(&chat).unwrap().anthropic;
        assert_eq!(anthropic.output_config.as_ref().unwrap().effort, "low");
        assert!(anthropic.thinking.is_some());

        // 各档位归一化后必须是后端 EffortTier 认得的值
        for (input, expected) in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
        ] {
            let chat = responses_to_chat_request(&build(input), Vec::new()).unwrap();
            let anthropic = chat_to_anthropic(&chat).unwrap().anthropic;
            assert_eq!(
                anthropic.output_config.as_ref().unwrap().effort,
                expected,
                "effort={input} 归一化错误"
            );
            assert!(anthropic.thinking.is_some(), "effort={input} 应开启 thinking");
        }
    }

    #[test]
    fn responses_request_expands_previous_messages() {
        let previous = vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(json!("first")),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }];
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4.5",
            "instructions": "stay terse",
            "input": "second"
        }))
        .unwrap();
        let chat = responses_to_chat_request(&req, previous).unwrap();
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[1].role, "system");
    }

    #[test]
    fn anthropic_response_becomes_openai_parts() {
        let parts = assistant_parts_from_anthropic(&json!({
            "model": "claude-sonnet-4.5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "call_1", "name": "run", "input": {"cmd": "ls"}}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": 2}
        }));
        assert_eq!(parts.text, "hello");
        assert_eq!(parts.tool_calls.len(), 1);
        assert_eq!(parts.cache_read_tokens, 2);
    }

    /// Codex 的 `ResponseCompletedUsage` 只认 Responses 口径的 usage 字段。
    /// 若 `response.completed` / 非流式响应体里带 chat 口径（prompt_tokens…），
    /// Codex 反序列化会失败并中断整条流。锁死字段名，防止回退到 usage_json。
    #[test]
    fn responses_object_uses_responses_usage_shape() {
        let parts = assistant_parts_from_anthropic(&json!({
            "model": "claude-sonnet-4.5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3
            }
        }));
        let obj = super::build_responses_object(
            "resp_1",
            123,
            None,
            None,
            &parts,
            &ToolKindMap::new(),
            &ResponsesResponseConfig::from_request(
                &serde_json::from_value(json!({"model": "gpt-5.6-sol", "input": "hi"}))
                    .unwrap(),
            ),
            "completed",
        );
        let usage = &obj["usage"];
        // Responses 口径字段必须存在
        assert_eq!(usage["input_tokens"], json!(15)); // 10 + 3 + 2
        assert_eq!(usage["output_tokens"], json!(5));
        assert_eq!(usage["total_tokens"], json!(20));
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], json!(2));
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], json!(0));
        // chat 口径字段绝不能出现
        assert!(usage.get("prompt_tokens").is_none());
        assert!(usage.get("completion_tokens").is_none());
    }

    /// 非流式 Responses 输出项必须符合 Codex 的 ResponseItem：
    /// message -> {type,role,content:[{type:"output_text",text}]}；
    /// function_call -> {type,name,arguments:<string>,call_id}。
    #[test]
    fn responses_object_output_items_match_codex_shape() {
        let parts = assistant_parts_from_anthropic(&json!({
            "model": "claude-sonnet-4.5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "done"},
                {"type": "tool_use", "id": "call_x", "name": "shell", "input": {"cmd": "ls"}}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let obj = super::build_responses_object(
            "resp_2",
            1,
            None,
            None,
            &parts,
            &ToolKindMap::new(),
            &ResponsesResponseConfig::from_request(
                &serde_json::from_value(json!({"model": "gpt-5.6-sol", "input": "hi"}))
                    .unwrap(),
            ),
            "completed",
        );
        let output = obj["output"].as_array().unwrap();
        let msg = output.iter().find(|i| i["type"] == "message").unwrap();
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"][0]["type"], "output_text");
        assert_eq!(msg["content"][0]["text"], "done");
        let fc = output.iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["name"], "shell");
        assert_eq!(fc["call_id"], "call_x");
        // arguments 必须是 JSON 字符串，而非对象
        assert!(fc["arguments"].is_string());
        assert_eq!(fc["arguments"], "{\"cmd\":\"ls\"}");
    }

    /// 端到端流式：把 Anthropic 的 text + tool_use SSE 喂给 ResponsesStreamTranslator，
    /// 验证输出的事件序列符合 Codex 解析器的要求：
    /// - 以 response.created / response.in_progress 开场；
    /// - message 的 output_item.done.item 可解析为 Codex ResponseItem::Message；
    /// - function_call 的 item 带字符串 arguments 与 call_id；
    /// - response.completed.response.usage 为 Responses 口径；
    /// - 以 data: [DONE] 收尾。
    #[test]
    fn responses_stream_emits_codex_compatible_events() {
        // 模拟 Kiro/Anthropic 上游发出的 SSE：先文本，再一个 tool_use，最后 usage。
        let upstream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"shell\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":12,\"output_tokens\":7,\"cache_read_input_tokens\":3}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );

        let events = run_responses_translator(upstream);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();

        // 开场
        assert_eq!(events[0].1["type"], "response.created");
        assert!(names.contains(&"response.in_progress"));
        // 文本增量存在
        assert!(names.contains(&"response.output_text.delta"));
        // 结尾是 [DONE]
        assert_eq!(events.last().unwrap().0, "[DONE]");

        // message item：能解析为 Codex ContentItem::OutputText 形状
        let msg_done = events
            .iter()
            .find(|(n, v)| n == "response.output_item.done" && v["item"]["type"] == "message")
            .expect("message output_item.done present");
        let item = &msg_done.1["item"];
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["content"][0]["type"], "output_text");
        assert_eq!(item["content"][0]["text"], "Hello");

        // function_call item：Codex ResponseItem::FunctionCall 形状
        let fc_done = events
            .iter()
            .find(|(n, v)| n == "response.output_item.done" && v["item"]["type"] == "function_call")
            .expect("function_call output_item.done present");
        let fc = &fc_done.1["item"];
        assert_eq!(fc["name"], "shell");
        assert_eq!(fc["call_id"], "toolu_1");
        assert!(fc["arguments"].is_string());
        assert_eq!(fc["arguments"], "{\"cmd\":\"ls\"}");

        // response.completed 的 usage 必须是 Responses 口径
        let completed = events
            .iter()
            .find(|(n, _)| n == "response.completed")
            .expect("response.completed present");
        let usage = &completed.1["response"]["usage"];
        assert_eq!(usage["input_tokens"], json!(15)); // 12 + 3
        assert_eq!(usage["output_tokens"], json!(7));
        assert_eq!(usage["total_tokens"], json!(22));
        assert!(usage.get("prompt_tokens").is_none());
    }

    /// chat completions 流式：纯文本。验证首个 chunk 带 role=assistant，
    /// 文本以 delta.content 增量下发，末尾 chunk 带 finish_reason=stop，
    /// 且不请求 usage 时不夹带 usage 对象，最后以 [DONE] 收尾。
    #[test]
    fn chat_stream_emits_openai_text_chunks() {
        let upstream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );

        let chunks = run_chat_translator(upstream, false);

        // 首个 chunk 声明 role，object 恒为 chat.completion.chunk
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");

        // 文本增量拼接完整
        let text: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "Hi there");

        // 收尾：倒数第二个 chunk 带 finish_reason，最后是 [DONE]
        assert_eq!(chunks.last().unwrap(), &Value::Null);
        let finish = chunks
            .iter()
            .find_map(|c| c["choices"][0]["finish_reason"].as_str());
        assert_eq!(finish, Some("stop"));

        // 未请求 include_usage 时不夹带 usage
        assert!(chunks.iter().all(|c| c["usage"].is_null()));
    }

    /// chat completions 流式：tool_use。验证 content_block_stop 时一次性发出
    /// 完整 tool_calls delta（index/id/function.name/arguments），
    /// finish_reason=tool_calls，且 include_usage 时末尾 chunk 带 OpenAI 口径 usage。
    #[test]
    fn chat_stream_emits_tool_calls_and_usage() {
        let upstream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"cache_read_input_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );

        let chunks = run_chat_translator(upstream, true);

        // tool_calls delta：一次性带全量 id/name/arguments
        let tool_delta = chunks
            .iter()
            .find(|c| c["choices"][0]["delta"].get("tool_calls").is_some())
            .expect("tool_calls delta present");
        let call = &tool_delta["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], json!(0));
        assert_eq!(call["id"], "toolu_1");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");

        // finish_reason 映射为 tool_calls
        let finish = chunks
            .iter()
            .find_map(|c| c["choices"][0]["finish_reason"].as_str());
        assert_eq!(finish, Some("tool_calls"));

        // include_usage：末尾带 OpenAI chat 口径 usage（prompt/completion/total）
        let usage = chunks
            .iter()
            .find_map(|c| (!c["usage"].is_null()).then(|| c["usage"].clone()))
            .expect("usage present when include_usage=true");
        assert_eq!(usage["prompt_tokens"], json!(12)); // 10 + 2 cache_read
        assert_eq!(usage["completion_tokens"], json!(4));
        assert_eq!(usage["total_tokens"], json!(16));
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], json!(2));
    }
}
