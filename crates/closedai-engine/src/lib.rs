// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference engine for closedAI. All inference flows through the [`Engine`]
//! trait so later milestones can substitute a distributed executor.

pub mod engine;
pub mod error;

pub use engine::{
    Engine, GeneratedToken, GenerationOutcome, GenerationParams, ModelInfo, StopReason,
};
pub use error::EngineError;
