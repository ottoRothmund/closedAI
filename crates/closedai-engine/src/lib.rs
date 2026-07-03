// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inference engine for closedAI. All inference flows through the [`Engine`]
//! trait so later milestones can substitute a distributed executor.

/// Temporary anchor so the workspace has a test before real types exist.
/// Removed in Task 2.
pub fn hello_ok() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_builds_and_tests_run() {
        assert!(hello_ok());
    }
}
