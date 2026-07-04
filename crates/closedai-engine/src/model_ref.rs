// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::PathBuf;

use thiserror::Error;

/// Where a model's weights come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    LocalPath(PathBuf),
    HuggingFace { repo: String, file: String },
}

#[derive(Debug, Error)]
pub enum ModelRefError {
    #[error("malformed model reference: {0}")]
    Malformed(String),
}

impl ModelRef {
    /// Parse a user-supplied model reference. `hf.co/<owner>/<repo>/<file.gguf>`
    /// is a Hugging Face reference; everything else is a local path.
    pub fn parse(input: &str) -> Result<ModelRef, ModelRefError> {
        let trimmed = input.trim();
        if let Some(rest) = trimmed.strip_prefix("hf.co/") {
            let parts: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
            if parts.len() < 3 {
                return Err(ModelRefError::Malformed(format!(
                    "expected hf.co/<owner>/<repo>/<file.gguf>, got '{input}'"
                )));
            }
            let file = parts[parts.len() - 1].to_string();
            let repo = parts[..parts.len() - 1].join("/");
            if !file.ends_with(".gguf") {
                return Err(ModelRefError::Malformed(format!(
                    "expected a .gguf file at the end of '{input}'"
                )));
            }
            return Ok(ModelRef::HuggingFace { repo, file });
        }
        Ok(ModelRef::LocalPath(PathBuf::from(trimmed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hugging_face_reference() {
        let r = ModelRef::parse(
            "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(
            r,
            ModelRef::HuggingFace {
                repo: "bartowski/SmolLM2-135M-Instruct-GGUF".to_string(),
                file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string(),
            }
        );
    }

    #[test]
    fn parses_local_path() {
        let r = ModelRef::parse("./models/tiny.gguf").unwrap();
        assert_eq!(r, ModelRef::LocalPath(PathBuf::from("./models/tiny.gguf")));
    }

    #[test]
    fn rejects_hf_reference_without_file() {
        let err = ModelRef::parse("hf.co/owner/repo");
        assert!(matches!(err, Err(ModelRefError::Malformed(_))));
    }
}
