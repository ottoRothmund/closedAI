// SPDX-License-Identifier: AGPL-3.0-or-later

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_closedai"))
}

#[test]
fn version_flag_prints_version() {
    let out = bin().arg("--version").output().expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("closedai"));
}

#[test]
fn run_requires_a_model() {
    let out = bin().arg("run").output().expect("run");
    assert!(!out.status.success(), "run without --model should fail");
}
