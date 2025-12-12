//! # Test Mode
//!
//! Control how tests handle numerical and compilation errors via the environment variable
//! `CUBE_TEST_MODE`.
//!
//! ## Modes
//! - `Correct` (default): Numerical errors fail the test, compilation errors are ignored.
//! - `Strict`: Both numerical and compilation errors fail the test.
//! - `Print { filter, only_failing }`:
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

use crate::test_utils::correctness::{TensorFilter, parse_tensor_filter};

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
    /// `only_failing = true`: only tests with numerical errors are marked as failed and printed.
    /// `only_failing = false`: all tests are marked as failed and printed.
    Print {
        filter: TensorFilter,
        only_failing: bool,
    },
}

impl TestMode {
    pub fn should_fail_on_test_compilation_fail(self) -> bool {
        match self {
            TestMode::Correct => false,
            TestMode::Strict => true,
            TestMode::Print { only_failing, .. } => !only_failing,
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
    } else {
        TestMode::Correct
    }
}

fn parse_print_mode(suffix: &str, only_failing: bool) -> TestMode {
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

    TestMode::Print {
        filter,
        only_failing,
    }
}
