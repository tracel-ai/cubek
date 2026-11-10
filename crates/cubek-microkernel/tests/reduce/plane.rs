//! The plane reduce microkernel alone: a whole plane reduces its row.

use cubek_microkernel::testing::reduce::ReduceScope;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{Reduce, ReduceCase};

/// A plane reduces every operation to the CPU's answer, rows shorter and longer than its lanes,
/// along the reduced axis and across it, each lane its own row.
#[test]
fn a_plane_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases = ReduceCase::every_operation(
        ReduceScope::Plane,
        &[(37, 1), (20, 4), (1024, 4)],
        &[(37, 1), (300, 4)],
    );
    Reduce { hw }.check(cases);
}
