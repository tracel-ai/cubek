//! Test-mode policy.
//!
//! Reads `cubek.toml` (see `crate::config`) once and decides whether each
//! test outcome should `Accept` or `Reject`. The TOML's `[test] policy`
//! field controls the pass/fail logic; the `[print]` section drives what
//! gets dumped to stdout (in `correctness::render`).

use crate::config::{TestPolicy, config};
use crate::{TestDecision, TestOutcome, ValidationResult};

/// Decide whether a test outcome passes or fails under the active policy.
pub fn decide(outcome: TestOutcome) -> TestDecision {
    use TestDecision::*;
    use TestOutcome::*;
    use ValidationResult::*;

    let cfg = config();
    let print_force_fail = cfg.print.enabled && cfg.print.force_fail;

    match cfg.test.policy {
        TestPolicy::Correct => match outcome {
            Validated(result) => match result {
                Pass => {
                    if print_force_fail {
                        Reject("print mode: tensors dumped".into())
                    } else {
                        Accept
                    }
                }
                Fail(reason) => Reject(reason),
                Error(reason) => Reject(reason),
                Skipped(_) => Accept,
            },
            CompileError(reason) => {
                if print_force_fail {
                    Reject(reason)
                } else {
                    Accept
                }
            }
        },
        TestPolicy::Strict => match outcome {
            Validated(result) => match result {
                Pass => {
                    if print_force_fail {
                        Reject("print mode: tensors dumped".into())
                    } else {
                        Accept
                    }
                }
                Fail(reason) => Reject(reason),
                Error(reason) => Reject(reason),
                Skipped(reason) => Reject(reason),
            },
            CompileError(reason) => Reject(reason),
        },
        TestPolicy::FailIfRun => match outcome {
            Validated(result) => match result {
                Pass => Reject("Actually passed, but fail-if-run policy active".into()),
                Fail(_) => Accept,
                Error(_) => Accept,
                Skipped(_) => Accept,
            },
            CompileError(_) => Accept,
        },
    }
}
