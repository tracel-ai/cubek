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
//!      - `Skip(String)`: test passes without checking anything, and says why on stderr.  
//!      - `Reject(String)`: test fails.  
//!    - Call [`TestDecision::enforce`] to actually fail the test.

use crate::test_mode::base::decide;
use cubecl::prelude::Client;
use cubecl::tile::launch::Decline;
use std::fmt::{Debug, Display};

/// CPU-only routines (e.g. `cpu_gemm`, where a plane is a thread) compute the wrong result
/// on a GPU backend. Records a `Skipped` outcome and returns `true` when `client` is not a
/// CPU backend (`num_cpu_cores` is `None`), so the caller can bail out of the test.
#[track_caller]
pub fn skip_unless_cpu(client: &Client) -> bool {
    if client.properties().hardware.num_cpu_cores.is_none() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "needs a CPU backend; this runtime reports no CPU cores".to_string(),
        ))
        .enforce();
        return true;
    }
    false
}

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
    Error(String),
    /// Validation was skipped. Useful to print stuff instead of actual testing
    Skipped(String),
}

/// Why a kernel gave no result to check against its reference.
#[derive(Debug)]
pub enum KernelResultError {
    /// Its routine refused the problem for this reason.
    Refused(Decline),
    /// It did not run for the reason given: it did not compile, or its inputs could not be built.
    NotRun(String),
}

impl Display for KernelResultError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Refused(why) => write!(f, "declined: {why}"),
            Self::NotRun(why) => f.write_str(why),
        }
    }
}

impl ExecutionOutcome {
    /// `Ok` where the kernel ran, else [`KernelResultError::NotRun`] with why it did not.
    pub fn into_kernel_result(self) -> Result<(), KernelResultError> {
        match self {
            Self::Executed => Ok(()),
            Self::CompileError(why) => Err(KernelResultError::NotRun(why)),
        }
    }
}

impl From<String> for KernelResultError {
    fn from(why: String) -> Self {
        Self::NotRun(why)
    }
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
    /// The outcome of a launch of `case` its scope refused for `why`: a skip, or a failure where
    /// the scope claims the launch and finds it broken ([`Decline::Broken`]).
    pub fn refused(case: &impl Debug, why: &Decline) -> Self {
        let reason = format!("{case:?}: {why}");
        match why.is_broken() {
            true => TestOutcome::Validated(ValidationResult::Fail(reason)),
            false => TestOutcome::Validated(ValidationResult::Skipped(reason)),
        }
    }

    /// The outcome of a check of `case` whose kernel gave no result: [`refused`](Self::refused)
    /// for a refusal, else a compile error.
    pub fn unrun(case: &impl Debug, error: KernelResultError) -> Self {
        match error {
            KernelResultError::Refused(why) => Self::refused(case, &why),
            KernelResultError::NotRun(why) => TestOutcome::CompileError(why),
        }
    }

    /// Apply the current test mode to this outcome and fail the test if rejected.
    ///
    /// Convenience wrapper for `decide(self).enforce()`: applies the
    /// active test policy (from `cubek.toml`) to this outcome and fails the
    /// test if the decision is `Reject`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let outcome = assert_equals_approx(&actual, &expected, 0.001).as_test_outcome();
    /// outcome.enforce(); // panics if the active policy rejects it
    /// ```
    #[track_caller]
    pub fn enforce(self) {
        decide(self).enforce();
    }
}

#[derive(Debug)]
/// The final policy-based verdict of a test, after applying the test mode.
/// Determines whether the test should be considered passing or failing.
pub enum TestDecision {
    /// The test is accepted (passes)
    Accept,
    /// The test checked nothing, because the device cannot run it or its routine declines it:
    /// it passes, and reports the skip on stderr
    Skip(String),
    /// The test is rejected (fails)
    Reject(String),
}

impl TestDecision {
    /// Actually asserts the test according to the decision.
    /// Panics if the test is rejected.
    #[track_caller]
    pub fn enforce(self) {
        match self {
            TestDecision::Accept => {}
            TestDecision::Skip(reason) => report_skip(&reason),
            TestDecision::Reject(reason) => panic!("Test failed: {}", reason),
        }
    }
}

/// `reason` reported on the process's own stderr, which the test harness does not capture as it
/// does `eprintln!`, so a skip shows in a run that passes.
fn report_skip(reason: &str) {
    use std::io::Write;
    let test = std::thread::current();
    let line = format!("SKIPPED {}: {reason}\n", test.name().unwrap_or("a test"));
    // A skip that cannot be reported still passes.
    let _ = std::io::stderr().write_all(line.as_bytes());
}

impl ValidationResult {
    /// Convert a `ValidationResult` into a `TestOutcome`.
    pub fn as_test_outcome(self) -> TestOutcome {
        TestOutcome::Validated(self)
    }
}
