//! 上游元数据事件。
//!
//! Kiro 会通过 `metadataEvent` 报告内容过滤，而不是发送 assistant 文本。

use serde::Deserialize;

use crate::kiro::parser::error::ParseResult;
use crate::kiro::parser::frame::Frame;

use super::base::EventPayload;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataEvent {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_details: StopDetails,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StopDetails {
    #[serde(default)]
    pub refusal: Option<RefusalDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RefusalDetails {
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub explanation: Option<String>,
}

impl MetadataEvent {
    pub fn is_content_filtered(&self) -> bool {
        self.stop_reason
            .as_deref()
            .is_some_and(|reason| reason.eq_ignore_ascii_case("CONTENT_FILTERED"))
    }

    pub fn refusal_explanation(&self) -> Option<&str> {
        self.stop_details
            .refusal
            .as_ref()
            .and_then(|refusal| refusal.explanation.as_deref())
            .map(str::trim)
            .filter(|explanation| !explanation.is_empty())
    }

    pub fn refusal_category(&self) -> Option<&str> {
        self.stop_details
            .refusal
            .as_ref()
            .and_then(|refusal| refusal.category.as_deref())
    }
}

impl EventPayload for MetadataEvent {
    fn from_frame(frame: &Frame) -> ParseResult<Self> {
        Ok(serde_json::from_slice(&frame.payload)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::parser::header::{HeaderValue, Headers};

    #[test]
    fn parses_content_filtered_refusal() {
        let mut headers = Headers::new();
        headers.insert(
            ":event-type".to_string(),
            HeaderValue::String("metadataEvent".to_string()),
        );
        let frame = Frame {
            headers,
            payload: br#"{"stopDetails":{"refusal":{"category":"CYBER","explanation":"The selected model cannot continue this conversation."}},"stopReason":"CONTENT_FILTERED"}"#.to_vec(),
        };

        let event = MetadataEvent::from_frame(&frame).unwrap();
        assert!(event.is_content_filtered());
        assert_eq!(event.refusal_category(), Some("CYBER"));
        assert_eq!(
            event.refusal_explanation(),
            Some("The selected model cannot continue this conversation.")
        );
    }
}
