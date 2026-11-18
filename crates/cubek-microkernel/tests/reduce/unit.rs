//! The unit reduce microkernel alone: every unit reduces its own row.

use crate::launch::reduce::TestedMicrokernel;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{ReduceCase, ReduceMicrokernelScopeUnderTest};

/// A unit alone reduces every operation to the CPU's answer, along the reduced axis and across
/// it, each lane its own row.
#[test]
fn a_unit_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases = ReduceCase::every_operation(
        TestedMicrokernel::UnitDirect,
        &[(37, 1), (64, 4)],
        &[(37, 1), (300, 4)],
    );
    ReduceMicrokernelScopeUnderTest { hw }.check(cases);
}
