//! The unit reduce microkernel alone: a plane's first unit reduces its row.

use cubek_microkernel::testing::reduce::ReduceScope;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{Reduce, ReduceCase};

/// A unit alone reduces every operation to the CPU's answer, along the reduced axis and across
/// it, each lane its own row.
#[test]
fn a_unit_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases =
        ReduceCase::every_operation(ReduceScope::Unit, &[(37, 1), (64, 4)], &[(37, 1), (300, 4)]);
    Reduce { hw }.check(cases);
}
