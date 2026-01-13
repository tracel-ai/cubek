#[derive(Debug)]
pub enum ExecutionOutcome {
    Executed,
    CompileError(String),
}

#[derive(Debug)]
pub enum TestOutcome {
    Validated(ValidationResult),
    CompileError(String),
}

#[derive(Debug)]
pub enum ValidationResult {
    Pass,
    Fail(String),
    Skipped(String),
}

#[derive(Debug)]
pub enum TestDecision {
    Accept,
    Reject(String),
}

impl TestDecision {
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
