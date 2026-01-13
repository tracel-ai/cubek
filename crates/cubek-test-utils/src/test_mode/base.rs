//! # Test Mode
//!
//! Control how tests handle numerical and compilation errors via the environment variable
//! `CUBE_TEST_MODE`.
//!
//! ## Modes
//! - `Correct` (default): Numerical errors fail the test, compilation errors are ignored.
//! - `Strict`: Both numerical and compilation errors fail the test.
//! - `Print { filter, fail_only }`:
//!     - `printall:<filter>` — all tests fail and matching elements are printed.
//!     - `printfail:<filter>` — only numerical errors fail and matching elements are printed.
//!
//! ## Filter Expressions
//! - The filter is **optional**. If omitted, all elements of the tensor are included.
//! - When specified, it is a comma-separated list of dimensions, supporting:
//!     - `.` to indicate a wildcard (all indices along that dimension),  
//!     - `N` for a single index,  
//!     - `M-K` for a range of indices.
//! - Example for a 4D tensor: `.,.,10-20,30` selects all elements where the 3rd dimension
//!   is 10–20 and the 4th dimension is 30, any values for the first two dimensions.
//! - **Important:** The number of entries in the filter must match the rank of the tensor.
//!
//! ## Examples
//!
//! ```bash
//! # Default mode: only numerical errors fail
//! export CUBE_TEST_MODE=Correct
//!
//! # Strict mode: all errors fail
//! export CUBE_TEST_MODE=Strict
//!
//! # Print all elements (no filter specified)
//! export CUBE_TEST_MODE=PrintAll
//!
//! # Print all elements in a subset of dimensions
//! export CUBE_TEST_MODE=PrintAll:.,10-20
//!
//! # Print only failing numerical elements
//! export CUBE_TEST_MODE=PrintFail:.,10-20
//! ```

use crate::{
    TestDecision, TestOutcome, ValidationResult,
    correctness::{TensorFilter, parse_tensor_filter},
};

const CUBE_TEST_MODE_ENV: &str = "CUBE_TEST_MODE";

#[derive(Default, Debug, Clone)]
pub enum TestMode {
    #[default]
    /// Numerical errors cause the test to fail.
    /// Compilation errors are accepted (do not fail the test).
    Correct,

    /// Both numerical and compilation errors cause the test to fail.
    Strict,

    /// All tests can be printed according to the given `filter`.
    /// `fail_only = true`: only tests with numerical errors are marked as failed and printed.
    /// `fail_only = false`: all tests are marked as failed and printed.
    Print {
        filter: TensorFilter,
        fail_only: bool,
    },

    /// Fail only if the test successfully runs.
    /// Compilation failures are ignored.
    ///
    /// Helpful to isolate relevant tests
    FailIfRun,
}

impl TestMode {
    pub fn decide(&self, outcome: TestOutcome) -> TestDecision {
        use TestDecision::*;
        use TestMode::*;
        use TestOutcome::*;
        use ValidationResult::*;

        match self {
            Correct => match outcome {
                Validated(result) => match result {
                    Pass => Accept,
                    Fail(reason) => Reject(reason),
                    Skipped(reason) => Reject(reason),
                },
                CompileError(_) => Accept,
            },
            Strict => match outcome {
                Validated(result) => match result {
                    Pass => Accept,
                    Fail(reason) => Reject(reason),
                    Skipped(reason) => Reject(reason),
                },
                CompileError(reason) => Reject(reason),
            },
            Print {
                filter: _,
                fail_only,
            } => match outcome {
                Validated(result) => match result {
                    Pass => {
                        if *fail_only {
                            Accept
                        } else {
                            Reject("printed".into())
                        }
                    }
                    Fail(reason) => Reject(reason),
                    Skipped(reason) => Reject(reason),
                },

                CompileError(reason) => {
                    if *fail_only {
                        Accept
                    } else {
                        Reject(reason)
                    }
                }
            },
            FailIfRun => match outcome {
                Validated(result) => match result {
                    Pass => Reject("Actually passed, but FailIfRun mode activated".to_string()),
                    Fail(_) => Accept,
                    Skipped(_) => Accept,
                },
                CompileError(_) => Accept,
            },
        }
    }
}

pub fn current_test_mode() -> TestMode {
    let val = match std::env::var(CUBE_TEST_MODE_ENV) {
        Ok(v) => v.to_lowercase(),
        Err(_) => return TestMode::Correct,
    };

    if let Some(print_mode) = val.strip_prefix("printall") {
        parse_print_mode(print_mode, false)
    } else if let Some(print_mode) = val.strip_prefix("printfail") {
        parse_print_mode(print_mode, true)
    } else if val == "strict" {
        TestMode::Strict
    } else if val == "failifrun" {
        TestMode::FailIfRun
    } else {
        TestMode::Correct
    }
}

fn parse_print_mode(suffix: &str, fail_only: bool) -> TestMode {
    let filter = if let Some(rest) = suffix.strip_prefix(':') {
        match parse_tensor_filter(rest) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Invalid print filter '{}': {}", rest, e);
                vec![]
            }
        }
    } else {
        vec![]
    };

    TestMode::Print { filter, fail_only }
}
