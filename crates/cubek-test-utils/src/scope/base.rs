//! One shape for testing any scope: a kernel, or a microkernel launched alone. A test states its
//! cases, launches the scope on each, and says what the reference writes; [`ScopeChecker::check`]
//! compares the two at the case's tolerance and hands the outcome to the active test policy. A
//! scope that refuses a case on the device is a skip, which the policy decides on, never a pass;
//! a launch it accepted and the device failed is a failure, and so is one it claims and finds
//! broken ([`Decline::Broken`]).
//!
//! The reference is the scope's own: a microkernel's is written for what it computes alone, a
//! kernel's for what its routines compute, even where the two happen to agree.

use core::fmt::Debug;

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;

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
///
/// # Assumptions
///
/// - [`reference`](Self::reference) returns one value per output [`launch`](Self::launch) returns
///   for the same case, in its order.
///
/// # Guarantees
///
/// - A case the scope refuses as a [`Decline::Broken`] fails, and so does a case it launches whose
///   output differs from its reference or that the device then fails to run.
pub trait ScopeChecker {
    /// One problem the scope runs on, named in every failure.
    type Case: Debug;

    /// Launch the scope on `case` and return every output it writes.
    ///
    /// # Errors
    ///
    /// Why the device runs no such scope at `case`, or a [`Decline::Broken`] where the scope
    /// breaks what it claimed.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Decline>;

    /// What the scope must write for `case`, computed on the host.
    fn reference(&self, case: &Self::Case) -> Vec<HostData>;

    /// The largest difference from the reference accepted at `case`, relative to the reference's
    /// value, or absolute where that value is below one.
    fn tolerance(&self, case: &Self::Case) -> f32;

    /// The tolerance the output `name` is checked at for `case`: [`tolerance`](Self::tolerance),
    /// unless the scope holds one output to its own, as a state kept in `f32` beside an output
    /// rounded to a narrower element, or a buffer the scope must leave bit for bit.
    fn tolerance_of(&self, case: &Self::Case, _name: &'static str) -> f32 {
        self.tolerance(case)
    }

    /// Launch the scope on every case and fail the test, through the active policy, at the first
    /// case whose outcome it rejects: that rejection is the panic the test fails by. Each case the
    /// scope refuses for the device is reported as a skip.
    ///
    /// # Panics
    ///
    /// Where an Assumption of [`ScopeChecker`] is broken.
    #[track_caller]
    fn check(&self, cases: impl IntoIterator<Item = Self::Case>) {
        let client = cubecl::test_device().client();
        for case in cases {
            self.outcome(&client, &case).enforce();
        }
    }

    /// [`check`](Self::check), for cases most of which the device refuses, as every setting a
    /// routine lists: a case refused for the device is passed over without a word where another
    /// case launches, and one skip is reported where none does.
    ///
    /// # Panics
    ///
    /// Where an Assumption of [`ScopeChecker`] is broken.
    #[track_caller]
    fn check_some(&self, cases: impl IntoIterator<Item = Self::Case>) {
        let client = cubecl::test_device().client();
        let mut refused = Vec::new();
        let mut launched = 0;
        let outcomes = cases.into_iter().filter_map(|case| {
            self.attempt(&client, &case)
                .map_err(|why| refused.push(why))
                .ok()
        });
        for outcome in outcomes {
            launched += 1;
            outcome.enforce();
        }
        if launched == 0 {
            TestOutcome::Validated(ValidationResult::Skipped(format!(
                "no case launched; refused: {refused:?}"
            )))
            .enforce();
        }
    }

    /// The outcome of one case, its failure naming the case and the output.
    #[track_caller]
    fn outcome(&self, client: &Client, case: &Self::Case) -> TestOutcome {
        self.attempt(client, case)
            .unwrap_or_else(|why| TestOutcome::refused(case, &why))
    }

    /// The outcome of one case the scope launched, or why it refused the case for the device. A
    /// launch the device failed to compile, to give resources to or to run fails, since the scope
    /// promised what it could not keep.
    ///
    /// # Panics
    ///
    /// Where an Assumption of [`ScopeChecker`] is broken.
    #[track_caller]
    fn attempt(&self, client: &Client, case: &Self::Case) -> Result<TestOutcome, Decline> {
        let written = match self.launch(client, case) {
            Ok(written) => written,
            Err(broken) if broken.is_broken() => return Ok(TestOutcome::refused(case, &broken)),
            Err(refused) => return Err(refused),
        };
        let handles: Vec<_> = written.iter().map(|output| &output.tensor.handle).collect();
        if let ExecutionOutcome::CompileError(why) = ExecutionOutcome::of_outputs(client, &handles)
        {
            return Ok(TestOutcome::Validated(ValidationResult::Fail(format!(
                "{case:?}: the launch it accepted failed: {why}"
            ))));
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
                    return Ok(TestOutcome::Validated(ValidationResult::Fail(named(why))));
                }
                ValidationResult::Error(why) => {
                    return Ok(TestOutcome::Validated(ValidationResult::Error(named(why))));
                }
                skipped => return Ok(TestOutcome::Validated(skipped)),
            }
        }
        Ok(TestOutcome::Validated(ValidationResult::Pass))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scope that refuses its one case for the reason it holds.
    struct Refusing(Decline);

    impl ScopeChecker for Refusing {
        type Case = ();

        fn launch(&self, _client: &Client, _case: &()) -> Result<Vec<Written>, Decline> {
            Err(self.0.clone())
        }

        fn reference(&self, _case: &()) -> Vec<HostData> {
            Vec::new()
        }

        fn tolerance(&self, _case: &()) -> f32 {
            0.
        }
    }

    fn outcome(refusal: Decline) -> TestOutcome {
        Refusing(refusal).outcome(&cubecl::test_device().client(), &())
    }

    #[test]
    fn a_launch_the_scope_does_not_claim_is_a_skip() {
        assert!(matches!(
            outcome(Decline::NoUnits),
            TestOutcome::Validated(ValidationResult::Skipped(_))
        ));
    }

    #[test]
    fn a_check_whose_every_case_is_refused_is_a_skip() {
        Refusing(Decline::NoUnits).check_some([()]);
    }

    #[test]
    fn a_launch_the_scope_claims_and_finds_broken_fails() {
        let broken = Decline::broken_by("the cube holds units", Decline::NoUnits);

        assert!(matches!(
            outcome(broken),
            TestOutcome::Validated(ValidationResult::Fail(_))
        ));
    }
}
