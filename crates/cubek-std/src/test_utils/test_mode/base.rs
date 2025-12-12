use crate::test_utils::correctness::{TensorFilter, parse_tensor_filter};

const CUBEK_TEST_MODE_ENV: &str = "CUBEK_TEST_MODE";

#[derive(Default, Debug)]
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

pub fn current_test_mode() -> TestMode {
    let val = match std::env::var(CUBEK_TEST_MODE_ENV) {
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
