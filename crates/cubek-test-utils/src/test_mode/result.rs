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

impl From<ValidationResult> for TestOutcome {
    fn from(validated: ValidationResult) -> Self {
        TestOutcome::Validated(validated)
    }
}
