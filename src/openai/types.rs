use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::anthropic::types::{
    CacheControl, Message, MessagesRequest, OutputConfig, SystemMessage, Thinking, Tool,
};

pub const DEFAULT_OPENAI_COMPAT_MODEL: &str = "claude-sonnet-4.5";
const DEFAULT_MAX_TOKENS: i32 = 4096;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<i32>,
    pub max_completion_tokens: Option<i32>,
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub stream_options: Option<StreamOptions>,
    pub prompt_cache_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAIToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_function_type")]
    pub tool_type: String,
    pub function: OpenAIFunctionCall,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type", default = "default_function_type")]
    pub tool_type: String,
    pub function: Option<OpenAIToolFunction>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    pub namespace: Option<String>,
    pub format: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    pub instructions: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<OpenAITool>,
    pub tool_choice: Option<Value>,
    pub previous_response_id: Option<String>,
    pub store: Option<bool>,
    pub max_output_tokens: Option<i32>,
    pub max_tokens: Option<i32>,
    pub reasoning: Option<Value>,
    pub metadata: Option<Value>,
    pub prompt_cache_key: Option<String>,
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
}

fn default_parallel_tool_calls() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredToolKind {
    Function,
    Custom,
}

#[derive(Debug, Clone)]
pub struct DeclaredTool {
    pub kind: DeclaredToolKind,
    pub name: String,
    pub namespace: Option<String>,
}

pub type ToolKindMap = HashMap<String, DeclaredTool>;

#[derive(Debug)]
pub struct ConvertedOpenAIRequest {
    pub anthropic: MessagesRequest,
    pub openai_messages: Vec<OpenAIMessage>,
    pub tool_kinds: ToolKindMap,
}

#[derive(Debug)]
pub struct OpenAIConversionError {
    pub message: String,
}

impl std::fmt::Display for OpenAIConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpenAIConversionError {}

fn default_function_type() -> String {
    "function".to_string()
}

fn err(message: impl Into<String>) -> OpenAIConversionError {
    OpenAIConversionError {
        message: message.into(),
    }
}

pub fn openai_model_to_kiro_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return DEFAULT_OPENAI_COMPAT_MODEL.to_string();
    }

    let lower = trimmed.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna"
    ) {
        return lower;
    }

    let looks_openai_native = lower.starts_with("gpt-")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("codex");
    if looks_openai_native {
        DEFAULT_OPENAI_COMPAT_MODEL.to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn chat_to_anthropic(
    req: &ChatCompletionRequest,
) -> Result<ConvertedOpenAIRequest, OpenAIConversionError> {
    chat_to_anthropic_with_metadata(req, None)
}

pub fn chat_to_anthropic_with_metadata(
    req: &ChatCompletionRequest,
    metadata: Option<crate::anthropic::types::Metadata>,
) -> Result<ConvertedOpenAIRequest, OpenAIConversionError> {
    if req.messages.is_empty() {
        return Err(err("messages must contain at least one message"));
    }

    let model = openai_model_to_kiro_model(&req.model);
    let (system, messages) = split_chat_messages(&req.messages)?;
    if messages.is_empty() {
        return Err(err("messages must contain at least one non-system message"));
    }

    let (thinking, output_config) = openai_reasoning_to_anthropic(
        &model,
        req.reasoning_effort.as_deref(),
        req.reasoning.as_ref(),
    );

    let (mut tools, tool_kinds) = convert_openai_tools(&req.tools);
    let tool_choice = convert_tool_choice(req.tool_choice.as_ref());
    if tool_choice
        .as_ref()
        .is_some_and(|choice| choice.get("type").and_then(Value::as_str) == Some("none"))
    {
        tools = None;
    }
    // OpenAI/Codex 客户端带 web_search 时强制走 agentic loop：纯快速路径恒返回 SSE 且
    // 只吐原始 web_search_tool_result 块，OpenAI 层既无法解析（非流式 502）也无法合成答案。
    let force_web_search_loop = tools
        .as_ref()
        .is_some_and(|list| list.iter().any(|t| t.name == "web_search"));

    Ok(ConvertedOpenAIRequest {
        anthropic: MessagesRequest {
            model,
            max_tokens: req
                .max_completion_tokens
                .or(req.max_tokens)
                .unwrap_or(DEFAULT_MAX_TOKENS)
                .max(1),
            messages,
            stream: req.stream,
            system,
            tools,
            tool_choice,
            thinking,
            output_config,
            metadata,
            force_web_search_loop,
        },
        openai_messages: req.messages.clone(),
        tool_kinds,
    })
}

pub fn responses_to_chat_request(
    req: &ResponsesRequest,
    previous_messages: Vec<OpenAIMessage>,
) -> Result<ChatCompletionRequest, OpenAIConversionError> {
    let mut messages = previous_messages;
    if let Some(instructions) = &req.instructions {
        let text = content_to_text(instructions);
        if !text.trim().is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(Value::String(text)),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            });
        }
    }

    let input_messages = responses_input_to_messages(req.input.as_ref())?;
    messages = append_openai_messages(messages, input_messages);
    if messages.is_empty() {
        return Err(err("input must contain at least one message"));
    }

    let mut tools = req.tools.clone();
    if let Some(Value::Array(items)) = req.input.as_ref() {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools")
                && let Some(entries) = item.get("tools").and_then(Value::as_array)
            {
                for entry in entries {
                    let tool = serde_json::from_value::<OpenAITool>(entry.clone())
                        .map_err(|error| err(format!("invalid additional tool: {error}")))?;
                    tools.push(tool);
                }
            }
        }
    }

    Ok(ChatCompletionRequest {
        model: req
            .model
            .as_deref()
            .unwrap_or(DEFAULT_OPENAI_COMPAT_MODEL)
            .to_string(),
        messages,
        stream: req.stream,
        max_tokens: req.max_tokens,
        max_completion_tokens: req.max_output_tokens,
        tools,
        tool_choice: req.tool_choice.clone(),
        reasoning_effort: None,
        reasoning: req.reasoning.clone(),
        stream_options: None,
        prompt_cache_key: req.prompt_cache_key.clone(),
    })
}

pub fn responses_input_to_messages(
    input: Option<&Value>,
) -> Result<Vec<OpenAIMessage>, OpenAIConversionError> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };

    match input {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => {
            if text.trim().is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(Value::String(text.clone())),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                }])
            }
        }
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                out = append_openai_messages(out, response_input_item_to_messages(item)?);
            }
            Ok(out)
        }
        Value::Object(_) => response_input_item_to_messages(input),
        _ => Err(err("unsupported input shape")),
    }
}

fn response_input_item_to_messages(
    item: &Value,
) -> Result<Vec<OpenAIMessage>, OpenAIConversionError> {
    let Some(obj) = item.as_object() else {
        return Ok(Vec::new());
    };
    let typ = obj.get("type").and_then(Value::as_str).unwrap_or_default();
    let role = obj.get("role").and_then(Value::as_str).unwrap_or_default();

    match typ {
        "additional_tools" | "reasoning" | "web_search_call" | "compaction" => {
            Ok(Vec::new())
        }
        "message" => {
            let role = if role.is_empty() { "user" } else { role };
            Ok(vec![OpenAIMessage {
                role: role.to_string(),
                content: obj.get("content").cloned(),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            }])
        }
        "input_text" | "text" => Ok(text_item_to_user_message(obj)),
        "input_image" | "image" | "image_url" => Ok(vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(Value::Array(vec![item.clone()])),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]),
        "output_text" => Ok(vec![OpenAIMessage {
            role: "assistant".to_string(),
            content: Some(Value::String(
                obj.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]),
        "function_call" => {
            let id = first_string(obj, &["call_id", "id"]);
            let name = first_string(obj, &["name"]);
            let arguments = stringify_value(obj.get("arguments"));
            if !arguments.trim().is_empty()
                && serde_json::from_str::<Value>(&arguments).is_err()
            {
                return Err(err(format!(
                    "input function_call {} has invalid JSON arguments",
                    id.as_deref().unwrap_or_default()
                )));
            }
            Ok(vec![OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(Value::String(String::new())),
                tool_calls: vec![OpenAIToolCall {
                    id,
                    tool_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.unwrap_or_default(),
                        arguments,
                    },
                    namespace: first_string(obj, &["namespace"]),
                }],
                tool_call_id: None,
                name: None,
            }])
        }
        "custom_tool_call" => {
            let id = first_string(obj, &["call_id", "id"]);
            let name = first_string(obj, &["name"]);
            let input = stringify_value(obj.get("input"));
            Ok(vec![OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(Value::String(String::new())),
                tool_calls: vec![OpenAIToolCall {
                    id,
                    tool_type: "custom".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.unwrap_or_default(),
                        arguments: input,
                    },
                    namespace: first_string(obj, &["namespace"]),
                }],
                tool_call_id: None,
                name: None,
            }])
        }
        "function_call_output" | "custom_tool_call_output" | "tool_result" => Ok(vec![OpenAIMessage {
            role: "tool".to_string(),
            content: obj.get("output").cloned().or_else(|| obj.get("content").cloned()),
            tool_calls: Vec::new(),
            tool_call_id: first_string(obj, &["call_id", "tool_call_id"]),
            name: None,
        }]),
        _ if !role.is_empty() => Ok(vec![OpenAIMessage {
            role: role.to_string(),
            content: obj.get("content").cloned(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]),
        _ => Ok(Vec::new()),
    }
}

fn text_item_to_user_message(obj: &Map<String, Value>) -> Vec<OpenAIMessage> {
    let text = obj
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(Value::String(text)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }]
    }
}

pub fn append_openai_messages(
    mut out: Vec<OpenAIMessage>,
    messages: Vec<OpenAIMessage>,
) -> Vec<OpenAIMessage> {
    for msg in messages {
        let merge_tool_calls = out
            .last()
            .is_some_and(|last| {
                last.role == "assistant"
                    && !last.tool_calls.is_empty()
                    && msg.role == "assistant"
                    && !msg.tool_calls.is_empty()
                    && content_to_text(last.content.as_ref().unwrap_or(&Value::Null))
                        .trim()
                        .is_empty()
                    && content_to_text(msg.content.as_ref().unwrap_or(&Value::Null))
                        .trim()
                        .is_empty()
            });
        if merge_tool_calls {
            if let Some(last) = out.last_mut() {
                last.tool_calls.extend(msg.tool_calls);
            }
        } else {
            out.push(msg);
        }
    }
    out
}

fn split_chat_messages(
    messages: &[OpenAIMessage],
) -> Result<(Option<Vec<SystemMessage>>, Vec<Message>), OpenAIConversionError> {
    let mut system = Vec::new();
    let mut out = Vec::new();
    // 缓冲连续的 OpenAI `tool` 消息：Anthropic 要求同一 assistant 轮次的所有
    // tool_result 必须**合并进一条 user 消息**（且紧跟在发起 tool_use 的 assistant
    // 之后）。OpenAI 的并行工具调用会发多条独立 tool 消息，若逐条转成单独的 user
    // 消息会产生连续 user 消息、破坏 tool_use/tool_result 配对，上游报
    // 400 "tool_use and tool_result blocks must be correctly paired and ordered"。
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for msg in messages {
        // 遇到任何非 tool 消息前，先把缓冲的 tool_result 作为一条 user 消息落盘。
        if msg.role != "tool" && !pending_tool_results.is_empty() {
            out.push(Message {
                role: "user".to_string(),
                content: Value::Array(std::mem::take(&mut pending_tool_results)),
            });
        }

        match msg.role.as_str() {
            "system" | "developer" => {
                let text = content_to_text(msg.content.as_ref().unwrap_or(&Value::Null));
                if !text.trim().is_empty() {
                    system.push(SystemMessage {
                        text,
                        cache_control: None,
                    });
                }
            }
            "user" => out.push(Message {
                role: "user".to_string(),
                content: openai_content_to_anthropic(msg.content.as_ref())?,
            }),
            "assistant" => out.push(Message {
                role: "assistant".to_string(),
                content: assistant_content_to_anthropic(msg)?,
            }),
            "tool" => pending_tool_results.push(json!({
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id.clone().unwrap_or_default(),
                "content": tool_result_content(msg.content.as_ref())?
            })),
            other => {
                tracing::debug!("忽略不支持的 OpenAI message role: {}", other);
            }
        }
    }

    // 收尾：flush 末尾残留的 tool_result（对话以工具结果结束是常见的续跑场景）。
    if !pending_tool_results.is_empty() {
        out.push(Message {
            role: "user".to_string(),
            content: Value::Array(pending_tool_results),
        });
    }

    Ok(((!system.is_empty()).then_some(system), out))
}

fn assistant_content_to_anthropic(msg: &OpenAIMessage) -> Result<Value, OpenAIConversionError> {
    let mut blocks = content_to_text_blocks(msg.content.as_ref())?;
    for call in &msg.tool_calls {
        if call.tool_type != "function" && call.tool_type != "custom" {
            continue;
        }
        let input = if call.tool_type == "custom" {
            json!({ "input": call.function.arguments })
        } else {
            parse_tool_arguments(&call.function.arguments)
        };
        blocks.push(json!({
            "type": "tool_use",
            "id": call.id.clone().unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
            "name": flat_tool_name(call.namespace.as_deref(), &call.function.name),
            "input": input,
        }));
    }

    if blocks.is_empty() {
        Ok(Value::String(String::new()))
    } else {
        Ok(Value::Array(blocks))
    }
}

fn openai_content_to_anthropic(content: Option<&Value>) -> Result<Value, OpenAIConversionError> {
    let Some(content) = content else {
        return Ok(Value::String(String::new()));
    };
    match content {
        Value::String(_) => Ok(content.clone()),
        Value::Array(_) => {
            let blocks = content_to_text_blocks(Some(content))?;
            Ok(Value::Array(blocks))
        }
        Value::Null => Ok(Value::String(String::new())),
        other => Ok(Value::String(content_to_text(other))),
    }
}

fn content_to_text_blocks(content: Option<&Value>) -> Result<Vec<Value>, OpenAIConversionError> {
    let Some(content) = content else {
        return Ok(Vec::new());
    };

    match content {
        Value::String(text) => {
            if text.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![json!({"type": "text", "text": text})])
            }
        }
        Value::Array(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                append_content_part(&mut blocks, part)?;
            }
            Ok(blocks)
        }
        Value::Object(_) => {
            let mut blocks = Vec::new();
            append_content_part(&mut blocks, content)?;
            Ok(blocks)
        }
        Value::Null => Ok(Vec::new()),
        other => Ok(vec![json!({"type": "text", "text": other.to_string()})]),
    }
}

fn append_content_part(blocks: &mut Vec<Value>, part: &Value) -> Result<(), OpenAIConversionError> {
    let Some(obj) = part.as_object() else {
        return Ok(());
    };
    let typ = obj.get("type").and_then(Value::as_str).unwrap_or_default();
    match typ {
        "text" | "input_text" | "output_text" => {
            if let Some(text) = obj.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                blocks.push(json!({"type": "text", "text": text}));
            }
        }
        "image_url" | "input_image" | "image" => {
            if let Some(block) = image_part_to_anthropic(part)? {
                blocks.push(block);
            }
        }
        _ => {
            if let Some(text) = obj.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                blocks.push(json!({"type": "text", "text": text}));
            }
        }
    }
    Ok(())
}

fn image_part_to_anthropic(part: &Value) -> Result<Option<Value>, OpenAIConversionError> {
    let Some(obj) = part.as_object() else {
        return Ok(None);
    };
    let url = obj
        .get("image_url")
        .and_then(|v| {
            v.as_object()
                .and_then(|o| o.get("url").and_then(Value::as_str))
                .or_else(|| v.as_str())
        })
        .or_else(|| obj.get("url").and_then(Value::as_str));

    let Some(url) = url else {
        return Ok(None);
    };

    if let Some((media_type, data)) = parse_data_url(url) {
        return Ok(Some(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data
            }
        })));
    }

    Ok(Some(json!({
        "type": "text",
        "text": format!("[image_url: {}]", url)
    })))
}

fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (media_type, data) = rest.split_once(";base64,")?;
    Some((media_type, data))
}

fn tool_result_content(content: Option<&Value>) -> Result<Value, OpenAIConversionError> {
    let Some(content) = content else {
        return Ok(Value::String(String::new()));
    };

    match content {
        Value::String(_) => Ok(content.clone()),
        Value::Array(_) | Value::Object(_) => {
            let blocks = content_to_text_blocks(Some(content))?;
            if blocks.is_empty() {
                Ok(Value::String(content_to_text(content)))
            } else {
                Ok(Value::Array(blocks))
            }
        }
        Value::Null => Ok(Value::String(String::new())),
        other => Ok(Value::String(other.to_string())),
    }
}

/// OpenAI 内置 web search 工具的默认 max_uses（与 Anthropic 原生路径 websearch.rs 对齐）。
const DEFAULT_WEB_SEARCH_MAX_USES: i32 = 8;

/// 判断 OpenAI 工具是否为内置 web search 工具。
///
/// OpenAI/Codex 的 web search 工具类型有多个变体：`web_search`（Responses API 新版）、
/// `web_search_preview` / `web_search_preview_2025_03_11`（早期预览）。统一识别后
/// 归一化为 Anthropic 原生 `web_search_20250305`，否则会在 `tool_type != "function"`
/// 处被直接丢弃，导致 Codex 的联网搜索能力在 OpenAI 兼容层彻底失效。
fn is_openai_web_search_tool(tool_type: &str) -> bool {
    tool_type == "web_search" || tool_type.starts_with("web_search_")
}

pub fn flat_tool_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(namespace) if !namespace.is_empty() => format!("{namespace}__{name}"),
        _ => name.to_string(),
    }
}

fn convert_openai_tools(tools: &[OpenAITool]) -> (Option<Vec<Tool>>, ToolKindMap) {
    let mut out = Vec::new();
    let mut kinds = ToolKindMap::new();
    convert_openai_tool_entries(tools, None, &mut out, &mut kinds);
    ((!out.is_empty()).then_some(out), kinds)
}

fn convert_openai_tool_entries(
    tools: &[OpenAITool],
    namespace: Option<&str>,
    out: &mut Vec<Tool>,
    kinds: &mut ToolKindMap,
) {
    for tool in tools {
        // 内置 web search：转成 Anthropic 原生工具，交给后端 web_search 路由/agentic loop。
        if is_openai_web_search_tool(&tool.tool_type) {
            let max_uses = tool
                .parameters
                .as_ref()
                .and_then(|p| p.get("max_uses").or_else(|| p.get("max_num_results")))
                .and_then(Value::as_i64)
                .map(|v| v as i32)
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_WEB_SEARCH_MAX_USES);
            out.push(Tool {
                tool_type: Some("web_search_20250305".to_string()),
                name: "web_search".to_string(),
                description: String::new(),
                input_schema: BTreeMap::new(),
                max_uses: Some(max_uses),
                cache_control: None::<CacheControl>,
            });
            continue;
        }
        match tool.tool_type.as_str() {
            "namespace" => {
                let Some(name) = tool.name.as_deref().filter(|name| !name.trim().is_empty()) else {
                    continue;
                };
                if namespace.is_some() {
                    tracing::warn!("忽略嵌套 OpenAI namespace 工具组: {}", name);
                    continue;
                }
                let start = out.len();
                convert_openai_tool_entries(&tool.tools, Some(name), out, kinds);
                if let Some(description) = tool.description.as_deref().filter(|text| !text.is_empty()) {
                    for converted in &mut out[start..] {
                        converted.description = format!("[{name}] {description}\n{}", converted.description);
                    }
                }
            }
            "function" | "custom" => {
                let Some(function) = tool_function(tool) else {
                    continue;
                };
                let name = function.0.trim();
                if name.is_empty() {
                    continue;
                }
                let flat = flat_tool_name(namespace, name);
                let kind = if tool.tool_type == "custom" {
                    DeclaredToolKind::Custom
                } else {
                    DeclaredToolKind::Function
                };
                let mut description = function.1.unwrap_or_else(|| name.to_string());
                let input_schema = if kind == DeclaredToolKind::Custom {
                    if let Some(format) = &tool.format
                        && let Some(definition) = format.get("definition").and_then(Value::as_str)
                    {
                        let syntax = format
                            .get("syntax")
                            .and_then(Value::as_str)
                            .unwrap_or("grammar");
                        description.push_str(&format!("\n\nInput format ({syntax} grammar):\n{definition}"));
                    }
                    schema_to_btree(Some(json!({
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The complete raw tool input text. Do NOT wrap it in JSON or escape it."
                            }
                        },
                        "required": ["input"],
                        "additionalProperties": false
                    })))
                } else {
                    schema_to_btree(function.2)
                };
                kinds.insert(
                    flat.clone(),
                    DeclaredTool {
                        kind,
                        name: name.to_string(),
                        namespace: namespace.map(str::to_string),
                    },
                );
                out.push(Tool {
                    tool_type: None,
                    name: flat,
                    description,
                    input_schema,
                    max_uses: None,
                    cache_control: None::<CacheControl>,
                });
            }
            other => tracing::warn!(tool_type = %other, "忽略不支持的 OpenAI 工具类型"),
        }
    }
}

fn tool_function(tool: &OpenAITool) -> Option<(&str, Option<String>, Option<Value>)> {
    if let Some(function) = &tool.function {
        Some((
            function.name.as_str(),
            function.description.clone(),
            function.parameters.clone(),
        ))
    } else {
        Some((
            tool.name.as_deref()?,
            tool.description.clone(),
            tool.parameters.clone(),
        ))
    }
}

fn schema_to_btree(schema: Option<Value>) -> BTreeMap<String, Value> {
    let schema = schema.unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    match schema {
        Value::Object(map) => map.into_iter().collect(),
        _ => {
            let mut map = BTreeMap::new();
            map.insert("type".to_string(), Value::String("object".to_string()));
            map.insert("properties".to_string(), Value::Object(Map::new()));
            map
        }
    }
}

fn convert_tool_choice(choice: Option<&Value>) -> Option<Value> {
    let choice = choice?;
    if let Some(s) = choice.as_str() {
        return Some(match s {
            "none" => json!({"type": "none"}),
            "auto" => json!({"type": "auto"}),
            "required" => json!({"type": "any"}),
            _ => choice.clone(),
        });
    }
    let function_name = choice
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str);
    if let Some(name) = function_name {
        return Some(json!({"type": "tool", "name": name}));
    }
    if let Some(name) = choice.get("name").and_then(Value::as_str) {
        let namespace = choice.get("namespace").and_then(Value::as_str);
        return Some(json!({
            "type": "tool",
            "name": flat_tool_name(namespace, name),
        }));
    }
    Some(choice.clone())
}

fn openai_reasoning_to_anthropic(
    model: &str,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
) -> (Option<Thinking>, Option<OutputConfig>) {
    let effort = reasoning_effort
        .map(str::to_string)
        .or_else(|| {
            reasoning
                .and_then(|v| v.get("effort"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    let Some(effort) = effort else {
        return (None, None);
    };

    // Codex/OpenAI 的 effort 档位（none/minimal/low/medium/high/xhigh）归一化为后端
    // `EffortTier` 认得的值（low/medium/high/xhigh），并给出对应 thinking 预算：
    // - "none"：显式关闭推理，不下发 thinking / output_config。若原样透传，后端
    //   `EffortTier::parse("none")` 会失败并 fallback 到 high，等于把“关推理”变成高强度推理。
    // - "minimal"：后端无此档，降级到最低的 low（原样透传同样会被 parse 拒绝 → fallback high）。
    // - 其他未知值：兜底 medium。
    let is_gpt_5_6 = matches!(
        model.to_ascii_lowercase().as_str(),
        "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna"
    );
    if effort == "none" {
        return if is_gpt_5_6 {
            (
                None,
                Some(OutputConfig {
                    effort: "none".to_string(),
                }),
            )
        } else {
            (None, None)
        };
    }

    let (budget, normalized_effort) = match effort.as_str() {
        "minimal" | "low" => (4_000, "low"),
        "medium" => (12_000, "medium"),
        "high" => (20_000, "high"),
        "max" if is_gpt_5_6 => (20_000, "max"),
        "xhigh" | "max" => (20_000, "xhigh"),
        _ => (12_000, "medium"),
    };

    (
        Some(Thinking {
            thinking_type: "enabled".to_string(),
            budget_tokens: budget,
        }),
        Some(OutputConfig {
            effort: normalized_effort.to_string(),
        }),
    )
}

fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return json!({});
    }
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({ "arguments": arguments }))
}

pub fn content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(_) | Value::Bool(_) => content.to_string(),
        Value::Array(items) => items.iter().map(content_to_text).collect::<Vec<_>>().join(""),
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(Value::as_str) {
                return text.to_string();
            }
            if let Some(text) = obj.get("content").map(content_to_text)
                && !text.is_empty()
            {
                return text;
            }
            if let Some(output) = obj.get("output").map(content_to_text)
                && !output.is_empty()
            {
                return output;
            }
            String::new()
        }
    }
}

fn stringify_value(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn first_string(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[derive(Debug, Clone)]
pub struct AssistantParts {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<OpenAIToolCall>,
    pub stop_reason: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
    pub cache_creation_tokens: i64,
    pub model: String,
    pub web_searches: Vec<(String, String)>,
    pub reasoning_tokens: i64,
    pub credit_usage: Option<f64>,
    pub credit_unit: Option<String>,
    pub credit_unit_plural: Option<String>,
}

pub fn assistant_parts_from_anthropic(value: &Value) -> AssistantParts {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut web_searches = Vec::new();

    if let Some(items) = value.get("content").and_then(Value::as_array) {
        for item in items {
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "text" => {
                    if let Some(t) = item.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
                "thinking" => {
                    if let Some(t) = item.get("thinking").and_then(Value::as_str) {
                        reasoning.push_str(t);
                    }
                }
                "tool_use" => {
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let input = item.get("input").cloned().unwrap_or_else(|| json!({}));
                    tool_calls.push(OpenAIToolCall {
                        id: Some(id),
                        tool_type: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name,
                            arguments: serde_json::to_string(&input)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                        namespace: None,
                    });
                }
                "server_tool_use"
                    if item.get("name").and_then(Value::as_str) == Some("web_search") =>
                {
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let query = item
                        .pointer("/input/query")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    web_searches.push((id, query));
                }
                _ => {}
            }
        }
    }

    let usage = value.get("usage").unwrap_or(&Value::Null);
    AssistantParts {
        text,
        reasoning,
        tool_calls,
        stop_reason: value
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("end_turn")
            .to_string(),
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        cache_read_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        cache_creation_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_OPENAI_COMPAT_MODEL)
            .to_string(),
        web_searches,
        reasoning_tokens: usage
            .get("reasoning_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        credit_usage: usage.get("credit_usage").and_then(Value::as_f64),
        credit_unit: usage
            .get("credit_unit")
            .and_then(Value::as_str)
            .map(str::to_string),
        credit_unit_plural: usage
            .get("credit_unit_plural")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

pub fn finish_reason_from_anthropic(stop_reason: &str) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" | "model_context_window_exceeded" => "length",
        _ => "stop",
    }
}

pub fn chat_message_from_parts(parts: &AssistantParts) -> Value {
    let content = if parts.text.is_empty() && !parts.tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(parts.text.clone())
    };
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !parts.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(parts.reasoning.clone());
    }
    if !parts.tool_calls.is_empty() {
        message["tool_calls"] = json!(parts.tool_calls);
    }
    message
}

pub fn usage_json(parts: &AssistantParts) -> Value {
    let prompt_tokens = parts.input_tokens + parts.cache_creation_tokens + parts.cache_read_tokens;
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": parts.output_tokens,
        "total_tokens": prompt_tokens + parts.output_tokens,
        "prompt_tokens_details": {
            "cached_tokens": parts.cache_read_tokens,
        },
        "completion_tokens_details": {
            "reasoning_tokens": parts.reasoning_tokens,
        }
    });
    append_credit_usage(&mut usage, parts);
    usage
}

/// Responses API 口径的 usage。
///
/// 与 chat completions 的 `prompt_tokens` / `completion_tokens` 不同，Responses
/// 用 `input_tokens` / `output_tokens` / `total_tokens`。Codex 的 SSE 解析器
/// (`ResponseCompletedUsage`) 只认这套字段名，若 `response.completed` 里带了 chat
/// 口径的 usage，反序列化会失败并中断整条流，所以必须单独构造。
pub fn responses_usage_json(parts: &AssistantParts) -> Value {
    let input_tokens = parts.input_tokens + parts.cache_creation_tokens + parts.cache_read_tokens;
    let mut usage = json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": parts.cache_read_tokens,
        },
        "output_tokens": parts.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": parts.reasoning_tokens,
        },
        "total_tokens": input_tokens + parts.output_tokens,
    });
    append_credit_usage(&mut usage, parts);
    usage
}

fn append_credit_usage(usage: &mut Value, parts: &AssistantParts) {
    if let Some(value) = parts.credit_usage {
        usage["credit_usage"] = json!(value);
    }
    if let Some(value) = &parts.credit_unit {
        usage["credit_unit"] = json!(value);
    }
    if let Some(value) = &parts.credit_unit_plural {
        usage["credit_unit_plural"] = json!(value);
    }
}

pub fn openai_error(message: impl Into<String>, error_type: impl Into<String>) -> Value {
    json!({
        "error": {
            "message": message.into(),
            "type": error_type.into(),
            "param": null,
            "code": null
        }
    })
}

pub fn assistant_message_for_history(parts: &AssistantParts) -> OpenAIMessage {
    OpenAIMessage {
        role: "assistant".to_string(),
        content: Some(Value::String(parts.text.clone())),
        tool_calls: parts.tool_calls.clone(),
        tool_call_id: None,
        name: None,
    }
}

pub fn response_output_from_parts(parts: &AssistantParts) -> (Vec<Value>, String) {
    response_output_from_parts_with_tools(parts, &ToolKindMap::new())
}

pub fn response_output_from_parts_with_tools(
    parts: &AssistantParts,
    tool_kinds: &ToolKindMap,
) -> (Vec<Value>, String) {
    let mut output = Vec::new();
    if !parts.reasoning.is_empty() {
        output.push(json!({
            "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
            "type": "reasoning",
            "summary": [{
                "type": "summary_text",
                "text": parts.reasoning,
            }],
            "content": null,
        }));
    }
    for (id, query) in &parts.web_searches {
        output.push(json!({
            "id": id,
            "type": "web_search_call",
            "status": "completed",
            "action": { "type": "search", "query": query },
        }));
    }
    if !parts.text.is_empty() {
        output.push(json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": parts.text,
                "annotations": []
            }]
        }));
    }
    for call in &parts.tool_calls {
        output.push(response_tool_item(call, tool_kinds, "completed"));
    }
    (output, parts.text.clone())
}

pub fn response_tool_item(
    call: &OpenAIToolCall,
    tool_kinds: &ToolKindMap,
    status: &str,
) -> Value {
    let declared = tool_kinds.get(&call.function.name);
    let kind = declared
        .map(|declared| declared.kind)
        .unwrap_or(DeclaredToolKind::Function);
    let name = declared
        .map(|declared| declared.name.as_str())
        .unwrap_or(call.function.name.as_str());
    let namespace = declared.and_then(|declared| declared.namespace.as_deref());
    let item_id = match kind {
        DeclaredToolKind::Function => format!("fc_{}", uuid::Uuid::new_v4().simple()),
        DeclaredToolKind::Custom => format!("ctc_{}", uuid::Uuid::new_v4().simple()),
    };
    let mut item = match kind {
        DeclaredToolKind::Function => json!({
            "id": item_id,
            "type": "function_call",
            "status": status,
            "call_id": call.id.clone().unwrap_or_default(),
            "name": name,
            "arguments": call.function.arguments,
        }),
        DeclaredToolKind::Custom => json!({
            "id": item_id,
            "type": "custom_tool_call",
            "status": status,
            "call_id": call.id.clone().unwrap_or_default(),
            "name": name,
            "input": custom_input_text(&call.function.arguments),
        }),
    };
    if let Some(namespace) = namespace {
        item["namespace"] = Value::String(namespace.to_string());
    }
    item
}

pub fn custom_input_text(arguments: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    if let Some(input) = value.get("input").and_then(Value::as_str) {
        return input.to_string();
    }
    if let Some(object) = value.as_object()
        && object.len() == 1
        && let Some(value) = object.values().next().and_then(Value::as_str)
    {
        return value.to_string();
    }
    arguments.to_string()
}
