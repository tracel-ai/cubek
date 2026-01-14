//! Kernel Test Workflow
//!
//! 1. **Execution**  
//!    - Kernel runs or fails to compile [`ExecutionOutcome`].
//!      - `Executed`: ran (correctness not checked).  
//!      - `CompileError`: did not compile.
//!
//! 2. **Validation**  
//!    - Check correctness of the executed kernel [`ValidationResult`].
//!      - `Pass`: result matches reference.  
//!      - `Fail`: result incorrect.  
//!      - `Skipped`: could not decide.
//!
//! 3. **Test Outcome**  
//!    - Combines execution + validation [`TestOutcome`].
//!
//! 4. **Policy Decision**  
//!    - Applies test mode to decide if the test passes [`TestDecision`].
//!      - `Accept`: test passes.  
//!      - `Reject(String)`: test fails.  
//!    - Call [`TestDecision::enforce`] to actually fail the test.

use crate::current_test_mode;
use std::fmt::Display;

#[derive(Debug)]
/// Whether a kernel was executed (without regard to correctness)
/// or failed to compile.
pub enum ExecutionOutcome {
    /// The kernel was executed successfully (correctness not checked)
    Executed,
    /// The kernel could not compile
    CompileError(String),
}

#[derive(Debug)]
/// The result of correctness validation for a kernel execution.
pub enum ValidationResult {
    /// The kernel passed the correctness test
    Pass,
    /// The kernel failed the correctness test
    Fail(String),
    /// The correctness test could not determine pass/fail
    Skipped(String),
}

#[derive(Debug)]
/// The overall outcome of a test, combining execution and validation.
/// Either the kernel was validated or failed to compile.
pub enum TestOutcome {
    /// The kernel was executed and validation was performed
    Validated(ValidationResult),
    /// The kernel could not compile
    CompileError(String),
}

impl TestOutcome {
    /// Apply the current test mode to this outcome and fail the test if rejected.
    ///
    /// This is a convenience wrapper around
    /// `current_test_mode().decide(self).enforce()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let outcome = assert_equals_approx(&actual, &expected, 0.001).as_test_outcome();
    /// outcome.enforce(); // panics if TestMode rejects it
    /// ```
    pub fn enforce(self) {
        current_test_mode().decide(self).enforce();
    }
}

#[derive(Debug)]
/// The final policy-based verdict of a test, after applying the test mode.
/// Determines whether the test should be considered passing or failing.
pub enum TestDecision {
    /// The test is accepted (passes)
    Accept,
    /// The test is rejected (fails)
    Reject(String),
}

impl TestDecision {
    /// Actually asserts the test according to the decision.
    /// Panics if the test is rejected.
    pub fn enforce(self) {
        match self {
            TestDecision::Accept => {}
            TestDecision::Reject(reason) => panic!("Test failed: {}", reason),
        }
    }
}

impl ValidationResult {
    /// Convert a `ValidationResult` into a `TestOutcome`.
    pub fn as_test_outcome(self) -> TestOutcome {
        TestOutcome::Validated(self)
    }
}

impl<E: Display> From<Result<(), E>> for ExecutionOutcome {
    fn from(result: Result<(), E>) -> Self {
        match result {
            Ok(_) => ExecutionOutcome::Executed,
            Err(err) => ExecutionOutcome::CompileError(format!("Test did not run: {}", err)),
        }
    }
}
