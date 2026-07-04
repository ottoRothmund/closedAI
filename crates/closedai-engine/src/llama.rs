// SPDX-License-Identifier: AGPL-3.0-or-later

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::engine::{
    Engine, GeneratedToken, GenerationOutcome, GenerationParams, ModelInfo, StopReason,
};
use crate::error::EngineError;

/// Process-global llama.cpp backend. `LlamaBackend::init()` may be called only
/// once per process, so it is shared across every `LlamaEngine` and every thread
/// (notably the parallel integration tests).
static BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn backend() -> Arc<LlamaBackend> {
    BACKEND
        .get_or_init(|| Arc::new(LlamaBackend::init().expect("llama.cpp backend init")))
        .clone()
}

/// Single-node llama.cpp-backed engine.
pub struct LlamaEngine {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    n_ctx: u32,
    info: ModelInfo,
}

impl LlamaEngine {
    /// Load a GGUF model. `n_gpu_layers` = 0 forces CPU; a large value offloads
    /// all layers when a GPU backend feature is compiled in.
    pub fn load(
        model_path: &Path,
        n_ctx: u32,
        n_gpu_layers: u32,
    ) -> Result<LlamaEngine, EngineError> {
        let backend = backend();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| EngineError::Load(e.to_string()))?;

        let info = ModelInfo {
            name: model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
            n_ctx_train: model.n_ctx_train(),
        };

        Ok(LlamaEngine {
            backend,
            model,
            n_ctx,
            info,
        })
    }

    fn build_sampler(params: &GenerationParams) -> LlamaSampler {
        if params.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(params.top_p, 1),
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(params.seed),
            ])
        }
    }
}

impl Engine for LlamaEngine {
    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: &mut dyn FnMut(GeneratedToken) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome, EngineError> {
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let prompt_tokens = tokens.len();
        if prompt_tokens == 0 {
            return Err(EngineError::Tokenize("prompt produced no tokens".into()));
        }
        if prompt_tokens >= self.n_ctx as usize {
            return Err(EngineError::ContextOverflow {
                prompt: prompt_tokens,
                n_ctx: self.n_ctx as usize,
            });
        }

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| EngineError::Load(e.to_string()))?;

        // Feed the prompt; request logits only for the final token.
        let mut batch = LlamaBatch::new(prompt_tokens, 1);
        let last = prompt_tokens - 1;
        for (i, tok) in tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == last)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| EngineError::Decode(e.to_string()))?;

        let mut sampler = Self::build_sampler(params);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = prompt_tokens as i32;
        let mut generated = 0usize;
        let mut acc = String::new();
        let mut stop_reason = StopReason::MaxTokens;

        while generated < params.max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                stop_reason = StopReason::Eos;
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| EngineError::Decode(e.to_string()))?;

            generated += 1;
            if let ControlFlow::Break(()) = on_token(GeneratedToken {
                token_id: token.0,
                text: piece.clone(),
            }) {
                stop_reason = StopReason::Cancelled;
                break;
            }

            // Stop-string check on the accumulated tail.
            acc.push_str(&piece);
            if !params.stop.is_empty() && params.stop.iter().any(|s| acc.ends_with(s)) {
                stop_reason = StopReason::StopString;
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
        }

        Ok(GenerationOutcome {
            prompt_tokens,
            tokens_generated: generated,
            stop_reason,
        })
    }
}
