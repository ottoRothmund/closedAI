// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference engine for closedAI. All inference flows through the [`Engine`]
//! trait so later milestones can substitute a distributed executor.

pub mod download;
pub mod engine;
pub mod error;
pub mod llama;
pub mod model_ref;

pub use download::resolve_model;
pub use engine::{
    Engine, GeneratedToken, GenerationOutcome, GenerationParams, ModelInfo, StopReason,
};
pub use error::EngineError;
pub use llama::LlamaEngine;
pub use model_ref::{ModelRef, ModelRefError};
