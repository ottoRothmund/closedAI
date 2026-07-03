// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

/// Errors from loading a model or generating text.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("failed to load model: {0}")]
    Load(String),

    #[error("failed to tokenize prompt: {0}")]
    Tokenize(String),

    #[error("decode step failed: {0}")]
    Decode(String),

    #[error("prompt is {prompt} tokens but the context window is {n_ctx}")]
    ContextOverflow { prompt: usize, n_ctx: usize },

    #[error("failed to download model: {0}")]
    Download(String),
}
