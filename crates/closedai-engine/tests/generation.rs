// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration test against a tiny real GGUF. Set CLOSEDAI_TEST_MODEL to the
//! path of a small .gguf (see the plan's Task 4 for how to fetch one). The test
//! is skipped when the env var is unset so `cargo test` stays offline by default.

use std::ops::ControlFlow;
use std::path::PathBuf;

use closedai_engine::{Engine, GenerationParams, LlamaEngine, StopReason};

fn test_model() -> Option<PathBuf> {
    std::env::var_os("CLOSEDAI_TEST_MODEL").map(PathBuf::from)
}

#[test]
fn generates_tokens_from_a_prompt() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set CLOSEDAI_TEST_MODEL to a small .gguf to run");
        return;
    };

    let mut engine = LlamaEngine::load(&model, 2048, 0).expect("load model");

    let params = GenerationParams {
        max_tokens: 16,
        temperature: 0.0, // greedy → deterministic
        ..GenerationParams::default()
    };

    let mut collected = String::new();
    let mut count = 0usize;
    let outcome = engine
        .generate("The capital of France is", &params, &mut |tok| {
            collected.push_str(&tok.text);
            count += 1;
            ControlFlow::Continue(())
        })
        .expect("generate");

    assert!(count > 0, "expected at least one token");
    assert_eq!(outcome.tokens_generated, count);
    assert!(matches!(
        outcome.stop_reason,
        StopReason::Eos | StopReason::MaxTokens
    ));
    assert!(!collected.trim().is_empty(), "expected non-empty output");
}
