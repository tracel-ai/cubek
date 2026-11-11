//! One shape for testing any scope: a kernel, or a microkernel launched alone. A test states its
//! cases, launches the scope on each, and says what the reference writes; [`ScopeUnderTest::check`]
//! compares the two at the case's tolerance and hands the outcome to the active test policy. A
//! scope the device does not run at a case is a skip, which the policy decides on, never a pass.
//!
//! The reference is the scope's own: a microkernel's is written for what it computes alone, a
//! kernel's for what its routines compute, even where the two happen to agree.

use core::fmt::Debug;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;

use crate::{
    ExecutionOutcome, HostData, HostDataType, TestOutcome, ValidationResult, assert_equals_approx,
};

/// One output a scope wrote for a case: its name in a failure, the tensor, and the type it is read
/// back to the host as.
pub struct Written {
    pub name: &'static str,
    pub tensor: TensorHandle,
    pub host: HostDataType,
}

/// A kernel or a microkernel under test, against its own reference.
pub trait ScopeUnderTest {
    /// One problem the scope runs on, named in every failure.
    type Case: Debug;

    /// Launch the scope on `case` and return every output it writes.
    ///
    /// # Errors
    ///
    /// Why the device runs no such scope at `case`: a skip, not a failure.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal>;

    /// What the scope must write for `case`, computed on the host: one value per output
    /// [`launch`](Self::launch) returns, in its order.
    fn reference(&self, case: &Self::Case) -> Vec<HostData>;

    /// The largest difference from the reference accepted at `case`, relative to the reference's
    /// value, or absolute where that value is below one.
    fn tolerance(&self, case: &Self::Case) -> f32;

    /// The tolerance the output `name` is held to at `case`: [`tolerance`](Self::tolerance),
    /// unless the scope holds one output to its own, as a state kept in `f32` beside an output
    /// rounded to a narrower element, or a buffer the scope must leave bit for bit.
    fn tolerance_of(&self, case: &Self::Case, _name: &'static str) -> f32 {
        self.tolerance(case)
    }

    /// Launch the scope on every case and fail the test, through the active policy, at the first
    /// case whose output differs from the reference.
    #[track_caller]
    fn check(&self, cases: impl IntoIterator<Item = Self::Case>) {
        let client = cubecl::test_device().client();
        for case in cases {
            self.outcome(&client, &case).enforce();
        }
    }

    /// The outcome of one case, its failure naming the case and the output.
    #[track_caller]
    fn outcome(&self, client: &Client, case: &Self::Case) -> TestOutcome {
        let written = match self.launch(client, case) {
            Ok(written) => written,
            Err(why) => {
                return TestOutcome::Validated(ValidationResult::Skipped(format!(
                    "{case:?}: {why:?}"
                )));
            }
        };
        let handles: Vec<_> = written.iter().map(|output| &output.tensor.handle).collect();
        if let ExecutionOutcome::CompileError(why) = ExecutionOutcome::of_outputs(client, &handles)
        {
            return TestOutcome::CompileError(format!("{case:?}: {why}"));
        }
        let reference = self.reference(case);
        assert_eq!(
            written.len(),
            reference.len(),
            "{case:?}: the reference holds one value per output"
        );
        for (output, want) in written.into_iter().zip(reference) {
            let tolerance = self.tolerance_of(case, output.name);
            let got = HostData::from_tensor_handle(client, output.tensor, output.host);
            let named = |why| format!("{case:?}, {}: {why}", output.name);
            match assert_equals_approx(&got, &want, tolerance) {
                ValidationResult::Pass => {}
                ValidationResult::Fail(why) => {
                    return TestOutcome::Validated(ValidationResult::Fail(named(why)));
                }
                ValidationResult::Error(why) => {
                    return TestOutcome::Validated(ValidationResult::Error(named(why)));
                }
                skipped => return TestOutcome::Validated(skipped),
            }
        }
        TestOutcome::Validated(ValidationResult::Pass)
    }
}
