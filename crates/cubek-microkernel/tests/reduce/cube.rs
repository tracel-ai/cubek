//! The cube reduce microkernel alone: a cube of four planes reduces its row.

use cubek_microkernel::testing::reduce::ReduceScope;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{Reduce, ReduceCase};

/// A cube of four planes reduces every operation to the CPU's answer, rows shorter than its units
/// and several times longer, along the reduced axis and across it, each lane its own row.
#[test]
fn a_cube_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases = ReduceCase::every_operation(
        ReduceScope::Cube { planes: 4 },
        &[(37, 1), (4096, 4)],
        &[(37, 1), (300, 4)],
    );
    Reduce { hw }.check(cases);
}
