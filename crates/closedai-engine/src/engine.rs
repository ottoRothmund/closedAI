// SPDX-License-Identifier: AGPL-3.0-or-later

use std::ops::ControlFlow;

use crate::error::EngineError;

/// Sampling and length controls for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            seed: 0,
            stop: Vec::new(),
        }
    }
}

/// One decoded token handed to the streaming callback.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    pub token_id: i32,
    pub text: String,
}

/// Why a generation loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopString,
    Cancelled,
}

/// Result of a completed (or cancelled) generation call.
#[derive(Debug, Clone)]
pub struct GenerationOutcome {
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
}

/// Static facts about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub n_ctx_train: u32,
}

/// The one boundary all inference crosses. Implementations may be single-node
/// (Milestone 01) or distributed (later milestones); callers never know which.
pub trait Engine: Send {
    fn model_info(&self) -> &ModelInfo;

    /// Generate from `prompt`, invoking `on_token` for each decoded token.
    /// Returning `ControlFlow::Break` from the callback stops generation with
    /// `StopReason::Cancelled`.
    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut dyn FnMut(GeneratedToken) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_params_have_sane_defaults() {
        let p = GenerationParams::default();
        assert_eq!(p.max_tokens, 512);
        assert!((p.temperature - 0.8).abs() < f32::EPSILON);
        assert!((p.top_p - 0.95).abs() < f32::EPSILON);
        assert!(p.stop.is_empty());
    }

    #[test]
    fn stop_reason_is_equatable() {
        assert_eq!(StopReason::Eos, StopReason::Eos);
        assert_ne!(StopReason::Eos, StopReason::MaxTokens);
    }
}
