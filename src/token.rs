//! Token 计量模块。
//!
//! 客户端可见的输入用量优先由 Kiro `contextUsagePercentage` 与模型窗口
//! 直接换算。百分比缺失、为零或无效时，才回退到请求内容计数：
//! 优先使用配置的 `count_tokens` API，失败时用 `cl100k_base` 本地估算。
//! 按 Kiro-Go 的公开 usage 口径扣除 Kiro 后端固定系统上下文，但不再
//! 乘旧的 1.843 系数，以保持各输入档位与当前测试基准的线性一致。

use crate::anthropic::types::{
    CountTokensRequest, CountTokensResponse, Message, SystemMessage, Tool,
};
use crate::http_client::{ProxyConfig, build_client};
use crate::model::config::TlsBackend;
use std::sync::OnceLock;

const CLAUDE_TOKEN_CORRECTION_FACTOR: f64 = 1.10;
// 沿用 Kiro-Go 的公开 usage 校准结构：context occupancy 包含一段
// 不应计入客户端 input usage 的后端系统提示。常量基于 2026-07-20
// tokencheap.io 与两组 aitokentest 凭证的最新 R8 结果标定。
const KIRO_DEFAULT_SYSTEM_PROMPT_TOKENS: i32 = 6_504;
const CLAUDE_PUBLIC_CONTEXT_USAGE_MIN_TOKENS: i32 = 8;
const APPROX_IMAGE_INPUT_TOKENS: u64 = 100;
const APPROX_DOCUMENT_INPUT_TOKENS: u64 = 3_000;

static CL100K: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();

/// Count Tokens API 配置
#[derive(Clone, Default)]
pub struct CountTokensConfig {
    /// 外部 count_tokens API 地址
    pub api_url: Option<String>,
    /// count_tokens API 密钥
    pub api_key: Option<String>,
    /// count_tokens API 认证类型（"x-api-key" 或 "bearer"）
    pub auth_type: String,
    /// 代理配置
    pub proxy: Option<ProxyConfig>,

    pub tls_backend: TlsBackend,
}

/// 全局配置存储
static COUNT_TOKENS_CONFIG: OnceLock<CountTokensConfig> = OnceLock::new();

/// 初始化 count_tokens 配置
///
/// 应在应用启动时调用一次
pub fn init_config(config: CountTokensConfig) {
    let _ = COUNT_TOKENS_CONFIG.set(config);
}

/// 获取配置
fn get_config() -> Option<&'static CountTokensConfig> {
    COUNT_TOKENS_CONFIG.get()
}

fn cl100k_count(text: &str) -> Option<u64> {
    if text.is_empty() {
        return Some(0);
    }
    CL100K
        .get_or_init(|| match tiktoken_rs::cl100k_base() {
            Ok(bpe) => Some(bpe),
            Err(e) => {
                tracing::warn!("cl100k_base 加载失败，回退词法 token 估算: {}", e);
                None
            }
        })
        .as_ref()
        .map(|bpe| bpe.encode_ordinary(text).len() as u64)
}

/// `cl100k_base` 文本 token 数；编码器不可用时回退词法估算。
pub fn count_tokens(text: &str) -> u64 {
    cl100k_count(text).unwrap_or_else(|| estimate_approx_tokens(text) as u64)
}

/// Kiro-Go `estimateApproxTokens` 的 Rust 实现，用于输出与 cache profile。
pub(crate) fn estimate_approx_tokens(text: &str) -> i32 {
    if text.is_empty() {
        return 0;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 5 {
        return (((chars.len() as f64) / 3.0).ceil() as i32).max(1);
    }

    let mut lexical = 0i32;
    let mut ascii_word_len = 0usize;
    let flush_word = |len: &mut usize, total: &mut i32| {
        if *len == 0 {
            return;
        }
        *total += if *len <= 12 {
            1
        } else {
            ((*len as f64) / 6.0).ceil() as i32
        };
        *len = 0;
    };
    for c in chars {
        if c.is_ascii_alphabetic() {
            ascii_word_len += 1;
            continue;
        }
        flush_word(&mut ascii_word_len, &mut lexical);
        if !c.is_whitespace() {
            lexical += 1;
        }
    }
    flush_word(&mut ascii_word_len, &mut lexical);
    ((lexical as f64 * CLAUDE_TOKEN_CORRECTION_FACTOR).ceil() as i32).max(1)
}

/// 估算请求的输入 tokens
///
/// 优先调用远程 API，失败时回退到本地计算。
///
/// 全部按引用收参：本函数在首字关键路径上每轮被调用一次，按值收参会导致每轮把
/// 整段会话（messages/system/tools）深拷贝一份，随对话轮数线性放大首字延迟。
pub(crate) fn count_all_tokens(
    model: &str,
    system: Option<&[SystemMessage]>,
    messages: &[Message],
    tools: Option<&[Tool]>,
) -> u64 {
    // 检查是否配置了远程 API
    if let Some(config) = get_config() {
        if let Some(api_url) = &config.api_url {
            // 尝试调用远程 API
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(call_remote_count_tokens(
                    api_url, config, model, system, messages, tools,
                ))
            });

            match result {
                Ok(tokens) => {
                    tracing::debug!("远程 count_tokens API 返回: {}", tokens);
                    return tokens;
                }
                Err(e) => {
                    tracing::warn!("远程 count_tokens API 调用失败，回退到本地计算: {}", e);
                }
            }
        }
    }

    // 本地计算
    count_all_tokens_local(system, messages, tools)
}

/// 调用远程 count_tokens API
async fn call_remote_count_tokens(
    api_url: &str,
    config: &CountTokensConfig,
    model: &str,
    system: Option<&[SystemMessage]>,
    messages: &[Message],
    tools: Option<&[Tool]>,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let client = build_client(config.proxy.as_ref(), 300, config.tls_backend)?;

    // 构建请求体（远程 API 需要拥有所有权的请求体，仅此分支付一次 clone；
    // 未配置远程 API 的默认路径不进入此函数，无 clone 开销）。
    let request = CountTokensRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        system: system.map(|s| s.to_vec()),
        tools: tools.map(|t| t.to_vec()),
    };

    // 构建请求
    let mut req_builder = client.post(api_url);

    // 设置认证头
    if let Some(api_key) = &config.api_key {
        if config.auth_type == "bearer" {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        } else {
            req_builder = req_builder.header("x-api-key", api_key);
        }
    }

    // 发送请求
    let response = req_builder
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("API 返回错误状态: {}", response.status()).into());
    }

    let result: CountTokensResponse = response.json().await?;
    Ok(result.input_tokens as u64)
}

/// 本地计算请求的输入 tokens
fn count_all_tokens_local(
    system: Option<&[SystemMessage]>,
    messages: &[Message],
    tools: Option<&[Tool]>,
) -> u64 {
    let mut raw_total = 0u64;

    // 系统消息
    if let Some(system) = system {
        for msg in system {
            raw_total += count_tokens(&msg.text);
        }
    }

    // Claude 消息 envelope：每条 4 token，整个对话尾部 3 token。
    for msg in messages {
        raw_total += 4;
        raw_total += count_tokens(&msg.role);
        raw_total += estimate_value_tokens(&msg.content);
    }
    if !messages.is_empty() {
        raw_total += 3;
    }

    // 工具定义
    if let Some(tools) = tools {
        for tool in tools {
            raw_total += 4;
            raw_total += count_tokens(&tool.name);
            raw_total += count_tokens(&tool.description);
            let input_schema_json = serde_json::to_string(&tool.input_schema).unwrap_or_default();
            raw_total += count_tokens(&input_schema_json);
        }
    }

    ((raw_total as f64 * CLAUDE_TOKEN_CORRECTION_FACTOR) as u64).max(1)
}

/// 递归估算 Claude content value，覆盖 tool_use/tool_result/图片/文档。
fn estimate_value_tokens(value: &serde_json::Value) -> u64 {
    use serde_json::Value;
    match value {
        Value::Null => 0,
        Value::String(text) => count_tokens(text),
        Value::Array(items) => items.iter().map(estimate_value_tokens).sum(),
        Value::Object(map) => {
            let block_type = map.get("type").and_then(Value::as_str).unwrap_or("");
            match block_type {
                "text" => map
                    .get("text")
                    .and_then(Value::as_str)
                    .map(count_tokens)
                    .unwrap_or(0),
                "thinking" => map
                    .get("thinking")
                    .and_then(Value::as_str)
                    .map(count_tokens)
                    .unwrap_or(0),
                "tool_use" => {
                    let mut total = map
                        .get("id")
                        .and_then(Value::as_str)
                        .map(count_tokens)
                        .unwrap_or(0);
                    total += map
                        .get("name")
                        .and_then(Value::as_str)
                        .map(count_tokens)
                        .unwrap_or(0);
                    if let Some(input) = map.get("input") {
                        total += count_tokens(&serde_json::to_string(input).unwrap_or_default());
                    }
                    total
                }
                "tool_result" => {
                    let mut total = map
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .map(count_tokens)
                        .unwrap_or(0);
                    if let Some(is_error) = map.get("is_error") {
                        total += count_tokens(if is_error.as_bool() == Some(true) {
                            "true"
                        } else {
                            "false"
                        });
                    }
                    total + map.get("content").map(estimate_value_tokens).unwrap_or(0)
                }
                "image" | "image_url" | "input_image" => APPROX_IMAGE_INPUT_TOKENS,
                "document" | "input_file" | "file" => APPROX_DOCUMENT_INPUT_TOKENS,
                _ => {
                    let mut total = 0;
                    if let Some(text) = map.get("text").and_then(Value::as_str) {
                        total += count_tokens(text);
                    }
                    if let Some(thinking) = map.get("thinking").and_then(Value::as_str) {
                        total += count_tokens(thinking);
                    }
                    if let Some(content) = map.get("content") {
                        total += estimate_value_tokens(content);
                    }
                    if total > 0 {
                        total
                    } else {
                        count_tokens(&serde_json::to_string(value).unwrap_or_default())
                    }
                }
            }
        }
        _ => count_tokens(&serde_json::to_string(value).unwrap_or_default()),
    }
}

/// 估算输出 tokens（与 Kiro-Go 轻量词法口径一致）。
pub(crate) fn estimate_output_tokens(content: &[serde_json::Value]) -> i32 {
    let mut total = 0;

    for block in content {
        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
            total += estimate_approx_tokens(text);
        }
        if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str()) {
            total += estimate_approx_tokens(thinking);
        }
        if block.get("type").and_then(|v| v.as_str()) == Some("redacted_thinking") {
            total += 8;
        }
        if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
            if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                total += estimate_approx_tokens(name);
            }
            if let Some(input) = block.get("input") {
                let input_str = serde_json::to_string(input).unwrap_or_default();
                total += estimate_approx_tokens(&input_str);
            }
        }
    }

    total.max(1)
}

/// 生成客户端可见的 public total input。
///
/// Kiro 百分比是首要依据：先换算已占用上下文，再扣掉同轮输出。
/// 按 Kiro-Go 的公开 Claude usage 逻辑，还需扣除 Kiro 后端自有的固定
/// system prompt 开销；小请求保留公开 usage envelope 下限。请求计数
/// 只在百分比不可用时回退。
pub(crate) fn finalize_public_input_tokens(
    estimated_input_tokens: i32,
    context_usage_percentage: Option<f64>,
    context_window: i32,
    output_tokens: i32,
) -> i32 {
    let fallback = estimated_input_tokens.max(0);
    let Some(percentage) =
        context_usage_percentage.filter(|value| value.is_finite() && *value > 0.0)
    else {
        return fallback;
    };

    let window = context_window.max(0);
    let occupied = ((window as f64) * percentage / 100.0).round() as i32;
    let context_input = (occupied - output_tokens.max(0)).max(0);
    let client_visible = context_input - KIRO_DEFAULT_SYSTEM_PROMPT_TOKENS;
    client_visible
        .max(CLAUDE_PUBLIC_CONTEXT_USAGE_MIN_TOKENS)
        .clamp(0, window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimate_output_tokens_counts_thinking_blocks() {
        let with_thinking = estimate_output_tokens(&[json!({
            "type": "thinking",
            "thinking": "需要计入输出 token"
        })]);
        let text_only = estimate_output_tokens(&[json!({
            "type": "text",
            "text": ""
        })]);

        assert!(with_thinking > text_only);
    }

    #[test]
    fn estimate_output_tokens_counts_redacted_thinking() {
        let tokens = estimate_output_tokens(&[json!({
            "type": "redacted_thinking",
            "data": "encrypted"
        })]);

        assert!(tokens >= 8);
    }

    #[test]
    fn local_token_count_is_monotonic_across_old_boundaries() {
        assert!(count_tokens(&"word ".repeat(100)) >= count_tokens(&"word ".repeat(99)));
        assert!(count_tokens(&"word ".repeat(800)) >= count_tokens(&"word ".repeat(799)));
    }

    #[test]
    fn public_input_tokens_match_latest_tokencheap_r8_baseline() {
        let baseline = [32, 1_023, 9_945, 99_177, 495_762];
        let context_inputs = [
            [6_516, 7_548, 16_432, 105_618, 502_202],
            [6_510, 7_473, 16_385, 105_617, 502_201],
        ];

        for samples in context_inputs {
            for (context_input, expected) in samples.into_iter().zip(baseline) {
                let percentage = context_input as f64 / 10_000.0;
                let actual = finalize_public_input_tokens(1, Some(percentage), 1_000_000, 0);
                let tolerance = (expected as f64 * 0.10).max(5.0);
                assert!(
                    (actual - expected).abs() as f64 <= tolerance,
                    "context input {context_input}: actual={actual}, baseline={expected}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn public_input_tokens_fall_back_when_kiro_percentage_is_unavailable() {
        assert_eq!(finalize_public_input_tokens(123, None, 1_000_000, 1), 123);
        assert_eq!(
            finalize_public_input_tokens(123, Some(0.0), 1_000_000, 1),
            123
        );
        assert_eq!(
            finalize_public_input_tokens(123, Some(f64::NAN), 1_000_000, 1),
            123
        );
    }

    #[test]
    fn input_estimator_counts_tool_results_and_images() {
        let value = json!([
            {"type":"tool_result","tool_use_id":"toolu_1","content":"hello world"},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"x"}}
        ]);
        assert!(estimate_value_tokens(&value) > APPROX_IMAGE_INPUT_TOKENS);
    }
}
