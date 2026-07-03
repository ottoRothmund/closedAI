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

/// Reproduces FIX B: a prompt tokenizing to more than 512 tokens used to fail
/// with a mislabeled `Decode("insufficient space")` error because the prompt
/// batch was hard-coded to `LlamaBatch::new(512, 1)`. The batch is now sized
/// to `prompt_tokens`, so prompts up to `n_ctx` should generate normally.
#[test]
fn long_prompt_over_512_tokens_generates() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set CLOSEDAI_TEST_MODEL to a small .gguf to run");
        return;
    };

    let mut engine = LlamaEngine::load(&model, 4096, 0).expect("load model");

    let prompt = "hello world ".repeat(400);

    let params = GenerationParams {
        max_tokens: 8,
        temperature: 0.0, // greedy → deterministic
        ..GenerationParams::default()
    };

    let outcome = engine
        .generate(&prompt, &params, &mut |_| ControlFlow::Continue(()))
        .expect("generate");

    eprintln!(
        "long prompt tokenized to {} tokens (must exceed 512)",
        outcome.prompt_tokens
    );
    assert!(
        outcome.prompt_tokens > 512,
        "test prompt must tokenize to more than 512 tokens to exercise FIX B, got {}",
        outcome.prompt_tokens
    );
    assert!(
        outcome.tokens_generated > 0,
        "expected at least one generated token"
    );
}

/// Reproduces FIX A: an empty prompt used to panic on a `prompt_tokens - 1`
/// subtraction underflow whenever tokenization produced zero tokens (reachable
/// from the CLI with an empty prompt on a model without a BOS token). The
/// point of this test is that the call below returns — `Ok` or `Err` — rather
/// than panicking.
#[test]
fn empty_prompt_does_not_panic() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set CLOSEDAI_TEST_MODEL to a small .gguf to run");
        return;
    };

    let mut engine = LlamaEngine::load(&model, 2048, 0).expect("load model");

    let params = GenerationParams {
        max_tokens: 8,
        temperature: 0.0,
        ..GenerationParams::default()
    };

    let result = engine.generate("", &params, &mut |_| ControlFlow::Continue(()));
    match &result {
        Ok(outcome) => eprintln!(
            "empty prompt tokenized to {} tokens (Ok)",
            outcome.prompt_tokens
        ),
        Err(e) => eprintln!("empty prompt returned Err: {e}"),
    }
    assert!(
        matches!(result, Ok(_) | Err(_)),
        "generate must return, not panic, on an empty prompt"
    );
}

/// Verifies the pinned llama.cpp loads the flagship architecture family.
/// Set CLOSEDAI_ARCH_MODEL to an ABSOLUTE path to a small `deepseek2`/MLA GGUF
/// (for example a GLM-4.7-Flash quant, or the smallest available DeepSeek-family
/// GGUF). Must be absolute: `cargo test` runs this binary with its working
/// directory set to the package directory, not the repo root, so a relative
/// path will not resolve as expected.
#[test]
fn loads_flagship_architecture() {
    let Some(model) = std::env::var_os("CLOSEDAI_ARCH_MODEL").map(std::path::PathBuf::from) else {
        eprintln!("skipping: set CLOSEDAI_ARCH_MODEL to a deepseek2/MLA GGUF to run");
        return;
    };
    let engine = closedai_engine::LlamaEngine::load(&model, 2048, 0);
    assert!(
        engine.is_ok(),
        "pinned llama.cpp must load deepseek2/MLA: {:?}",
        engine.err()
    );
}
