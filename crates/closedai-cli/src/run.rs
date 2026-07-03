// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io::Write;
use std::ops::ControlFlow;
use std::time::Instant;

use clap::Args;
use closedai_engine::{resolve_model, Engine, GenerationParams, LlamaEngine, ModelRef};

#[derive(Args)]
pub struct RunArgs {
    /// Model reference: a local .gguf path or hf.co/<owner>/<repo>/<file.gguf>.
    #[arg(long)]
    model: String,

    /// Prompt. If omitted, read a single line from stdin.
    prompt: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    #[arg(long, default_value_t = 0)]
    seed: u32,

    /// Context window size.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,

    /// Layers to offload to GPU (needs a GPU cargo feature). 0 = CPU only.
    #[arg(long, default_value_t = 0)]
    gpu_layers: u32,
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    let model_ref = ModelRef::parse(&args.model)?;
    let path = resolve_model(&model_ref)?;

    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            line.trim_end().to_string()
        }
    };

    let mut engine = LlamaEngine::load(&path, args.ctx, args.gpu_layers)?;
    let params = GenerationParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
        stop: Vec::new(),
    };

    let start = Instant::now();
    let mut stdout = std::io::stdout();
    let outcome = engine.generate(&prompt, &params, &mut |tok| {
        let _ = stdout.write_all(tok.text.as_bytes());
        let _ = stdout.flush();
        ControlFlow::Continue(())
    })?;
    let elapsed = start.elapsed();

    let tok_per_s = outcome.tokens_generated as f64 / elapsed.as_secs_f64().max(1e-9);
    eprintln!(
        "\n[{} prompt tokens, {} generated, {:.1} tok/s, {}]",
        outcome.prompt_tokens,
        outcome.tokens_generated,
        tok_per_s,
        format_stop(outcome.stop_reason),
    );
    Ok(())
}

fn format_stop(reason: closedai_engine::StopReason) -> &'static str {
    use closedai_engine::StopReason::*;
    match reason {
        Eos => "eos",
        MaxTokens => "max_tokens",
        StopString => "stop_string",
        Cancelled => "cancelled",
    }
}
