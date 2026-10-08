//! `/v1/models` 模型目录：自动同步 Kiro 上游 `ListAvailableModels`
//!
//! - 上游模型列表按 TTL 缓存在进程内，过期后由下一次 `/v1/models` 请求触发刷新
//! - 上游返回的 `claude-*` 模型自动补充 `-thinking` 变体
//! - 内置静态列表中的 Anthropic 风格别名（如 `claude-opus-4-7`、带日期的 ID）
//!   仅在其映射目标出现在上游列表时保留
//! - 上游不可用（无凭据 / 全部失败）时沿用上次缓存，若从未成功则回退到静态列表

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::kiro::model::available_models::UpstreamModel;
use crate::kiro::token_manager::MultiTokenManager;

use super::converter::normalize_model_id;
use super::types::Model;

/// 上游模型列表缓存有效期
const CACHE_TTL: Duration = Duration::from_secs(10 * 60);
/// 刷新失败后的重试间隔，避免上游故障时每个请求都打一次上游
const RETRY_INTERVAL: Duration = Duration::from_secs(60);
/// 单次刷新最多尝试的凭据数
const MAX_CREDENTIAL_ATTEMPTS: usize = 3;
/// 单个凭据查询超时
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
/// 上游未提供时的默认 max_tokens（与静态列表保持一致）
const DEFAULT_MAX_TOKENS: i32 = 64000;

#[derive(Default)]
struct CacheState {
    models: Option<Vec<UpstreamModel>>,
    /// 最近一次成功刷新时间
    refreshed_at: Option<Instant>,
    /// 最近一次尝试刷新时间（含失败）
    attempted_at: Option<Instant>,
}

fn cache() -> &'static RwLock<CacheState> {
    static CACHE: OnceLock<RwLock<CacheState>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(CacheState::default()))
}

/// 刷新互斥锁：同一时刻只允许一个请求去上游拉取
fn refresh_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn needs_refresh(state: &CacheState, now: Instant) -> bool {
    let fresh = state
        .refreshed_at
        .is_some_and(|t| now.duration_since(t) < CACHE_TTL);
    let recently_attempted = state
        .attempted_at
        .is_some_and(|t| now.duration_since(t) < RETRY_INTERVAL);
    !fresh && !recently_attempted
}

/// 获取（必要时刷新）上游模型列表；从未成功时返回 None
async fn upstream_models(token_manager: &Arc<MultiTokenManager>) -> Option<Vec<UpstreamModel>> {
    if !needs_refresh(&cache().read(), Instant::now()) {
        return cache().read().models.clone();
    }

    let _guard = refresh_lock().lock().await;
    // 等锁期间可能已被其它请求刷新
    if !needs_refresh(&cache().read(), Instant::now()) {
        return cache().read().models.clone();
    }

    let fetched = fetch_upstream(token_manager).await;
    let mut state = cache().write();
    let now = Instant::now();
    state.attempted_at = Some(now);
    if let Some(models) = fetched {
        state.models = Some(models);
        state.refreshed_at = Some(now);
    }
    state.models.clone()
}

async fn fetch_upstream(token_manager: &Arc<MultiTokenManager>) -> Option<Vec<UpstreamModel>> {
    let ids: Vec<u64> = token_manager
        .snapshot()
        .entries
        .iter()
        .filter(|e| !e.disabled)
        .map(|e| e.id)
        .take(MAX_CREDENTIAL_ATTEMPTS)
        .collect();

    for id in ids {
        match tokio::time::timeout(FETCH_TIMEOUT, token_manager.get_available_models_for(id)).await
        {
            Ok(Ok(resp)) if !resp.models.is_empty() => {
                tracing::info!(
                    credential_id = id,
                    count = resp.models.len(),
                    "已从 Kiro ListAvailableModels 同步模型列表"
                );
                return Some(resp.models);
            }
            Ok(Ok(_)) => tracing::warn!(credential_id = id, "ListAvailableModels 返回空列表"),
            Ok(Err(e)) => tracing::warn!(credential_id = id, error = %e, "ListAvailableModels 查询失败"),
            Err(_) => tracing::warn!(credential_id = id, "ListAvailableModels 查询超时"),
        }
    }
    None
}

fn infer_owner(model_id: &str) -> &'static str {
    let lower = model_id.to_ascii_lowercase();
    if lower.starts_with("claude-") {
        "anthropic"
    } else if lower.starts_with("gpt-") {
        "openai"
    } else if lower.starts_with("deepseek") {
        "deepseek"
    } else if lower.starts_with("minimax") {
        "minimax"
    } else if lower.starts_with("glm") {
        "zhipu"
    } else if lower.starts_with("qwen") {
        "qwen"
    } else {
        "kiro"
    }
}

/// 把上游 Claude 模型 ID 中带点的版本号转成短横线形式
///
/// 如 `claude-opus-5.5` → `claude-opus-5-5`；请求时 `normalize_model_id` 会再映射回
/// 上游 ID，因此对外暴露短横线形式不影响转发。
fn dash_claude_version(model_id: &str) -> String {
    let mut parts: Vec<&str> = model_id.split('-').collect();
    if parts.len() < 3 || !parts[0].eq_ignore_ascii_case("claude") {
        return model_id.to_string();
    }
    let is_dotted_version = parts[2].split_once('.').is_some_and(|(major, minor)| {
        !major.is_empty()
            && !minor.is_empty()
            && major.chars().all(|c| c.is_ascii_digit())
            && minor.chars().all(|c| c.is_ascii_digit())
    });
    if !is_dotted_version {
        return model_id.to_string();
    }
    let version = parts[2].replace('.', "-");
    parts[2] = &version;
    parts.join("-")
}

/// 合并上游模型与静态列表
///
/// 输出顺序：上游模型（含 thinking 变体）在前，随后是映射目标仍在上游列表中的静态别名。
pub(crate) fn merge_models(upstream: &[UpstreamModel], static_models: Vec<Model>) -> Vec<Model> {
    let upstream_ids: HashSet<String> = upstream
        .iter()
        .map(|m| normalize_model_id(&m.model_id))
        .collect();

    let static_for = |id: &str| {
        let target = normalize_model_id(id);
        static_models
            .iter()
            .find(|m| m.id == id)
            .or_else(|| static_models.iter().find(|m| normalize_model_id(&m.id) == target))
    };

    let mut models = Vec::new();
    for up in upstream {
        let base = static_for(&up.model_id);
        let display_name = up
            .model_name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .or_else(|| base.map(|m| m.display_name.clone()))
            .unwrap_or_else(|| up.model_id.clone());
        let model = Model {
            id: dash_claude_version(&up.model_id),
            object: "model".to_string(),
            created: base.map(|m| m.created).unwrap_or(0),
            owned_by: base
                .map(|m| m.owned_by.clone())
                .unwrap_or_else(|| infer_owner(&up.model_id).to_string()),
            display_name,
            model_type: "chat".to_string(),
            max_tokens: base.map(|m| m.max_tokens).unwrap_or(DEFAULT_MAX_TOKENS),
        };
        if up.model_id.to_ascii_lowercase().starts_with("claude-")
            && !up.model_id.ends_with("-thinking")
        {
            let thinking = Model {
                id: format!("{}-thinking", model.id),
                display_name: format!("{} (Thinking)", model.display_name),
                ..model.clone()
            };
            models.push(model);
            models.push(thinking);
        } else {
            models.push(model);
        }
    }

    models.extend(
        static_models
            .into_iter()
            .filter(|m| upstream_ids.contains(&normalize_model_id(&m.id))),
    );

    let mut seen = HashSet::new();
    models.retain(|m| seen.insert(m.id.clone()));
    models
}

/// 构建 `/v1/models` 返回的模型列表
pub(crate) async fn list_models(
    token_manager: Option<&Arc<MultiTokenManager>>,
    static_models: Vec<Model>,
) -> Vec<Model> {
    let Some(token_manager) = token_manager else {
        return static_models;
    };
    match upstream_models(token_manager).await {
        Some(upstream) if !upstream.is_empty() => merge_models(&upstream, static_models),
        _ => static_models,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn up(id: &str, name: Option<&str>) -> UpstreamModel {
        UpstreamModel {
            model_id: id.to_string(),
            model_name: name.map(str::to_string),
            description: None,
            token_limits: None,
        }
    }

    fn ids(models: &[Model]) -> Vec<&str> {
        models.iter().map(|m| m.id.as_str()).collect()
    }

    #[test]
    fn dash_claude_version_converts_dotted_versions() {
        assert_eq!(dash_claude_version("claude-opus-5.5"), "claude-opus-5-5");
        assert_eq!(dash_claude_version("claude-sonnet-5.5"), "claude-sonnet-5-5");
        assert_eq!(dash_claude_version("claude-haiku-5.5"), "claude-haiku-5-5");
        assert_eq!(
            dash_claude_version("claude-opus-5.5-thinking"),
            "claude-opus-5-5-thinking"
        );
        assert_eq!(dash_claude_version("claude-opus-5"), "claude-opus-5");
        assert_eq!(dash_claude_version("gpt-5.6-sol"), "gpt-5.6-sol");
        assert_eq!(dash_claude_version("auto"), "auto");
    }

    #[test]
    fn merge_exposes_dashed_ids_that_map_back_to_upstream() {
        let upstream = vec![
            up("claude-opus-5.5", Some("Claude Opus 5.5")),
            up("claude-sonnet-5.5", None),
            up("claude-haiku-5.5", None),
        ];
        let merged = merge_models(&upstream, super::super::handlers::available_models());
        let ids = ids(&merged);

        for (dashed, upstream_id) in [
            ("claude-opus-5-5", "claude-opus-5.5"),
            ("claude-sonnet-5-5", "claude-sonnet-5.5"),
            ("claude-haiku-5-5", "claude-haiku-5.5"),
        ] {
            assert!(ids.contains(&dashed), "missing {dashed}");
            let thinking = format!("{dashed}-thinking");
            assert!(ids.contains(&thinking.as_str()), "missing {thinking}");
            assert_eq!(normalize_model_id(dashed), upstream_id);
            assert_eq!(normalize_model_id(&thinking), upstream_id);
        }
        let opus = merged.iter().find(|m| m.id == "claude-opus-5-5").unwrap();
        assert_eq!(opus.display_name, "Claude Opus 5.5");
    }

    #[test]
    fn merge_adds_new_upstream_models_with_thinking_variant() {
        let upstream = vec![up("claude-opus-6", Some("Claude Opus 6")), up("auto", None)];
        let merged = merge_models(&upstream, super::super::handlers::available_models());
        let ids = ids(&merged);

        assert!(ids.contains(&"claude-opus-6"));
        assert!(ids.contains(&"claude-opus-6-thinking"));
        assert!(ids.contains(&"auto"));
        assert!(!ids.contains(&"auto-thinking"));
        let opus6 = merged.iter().find(|m| m.id == "claude-opus-6").unwrap();
        assert_eq!(opus6.owned_by, "anthropic");
        assert_eq!(opus6.display_name, "Claude Opus 6");
    }

    #[test]
    fn merge_keeps_only_aliases_supported_upstream() {
        let upstream = vec![up("claude-opus-4.7", None)];
        let merged = merge_models(&upstream, super::super::handlers::available_models());
        let ids = ids(&merged);

        // 上游 ID 转为短横线形式；静态列表中的点号别名仍保留
        assert!(ids.contains(&"claude-opus-4-7"));
        assert!(ids.contains(&"claude-opus-4-7-thinking"));
        assert!(ids.contains(&"claude-opus-4.7"));
        assert!(!ids.contains(&"claude-opus-4-8"));
        assert!(!ids.contains(&"claude-sonnet-5"));
        // 静态列表中的展示名作为上游缺省时的回退
        let m = merged.iter().find(|m| m.id == "claude-opus-4-7").unwrap();
        assert_eq!(m.display_name, "Claude Opus 4.7");
    }

    #[test]
    fn merge_has_no_duplicate_ids() {
        let upstream = vec![
            up("claude-opus-5.5", None),
            up("claude-opus-5.5-thinking", None),
        ];
        let merged = merge_models(&upstream, super::super::handlers::available_models());
        let ids = ids(&merged);
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
        assert!(ids.contains(&"claude-opus-5-5"));
    }

    #[tokio::test]
    async fn list_models_without_provider_falls_back_to_static() {
        let models = list_models(None, super::super::handlers::available_models()).await;
        assert_eq!(models.len(), super::super::handlers::available_models().len());
    }
}
