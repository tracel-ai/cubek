//! The cube reduce microkernel alone: a cube of four planes reduces its row.

use crate::launch::reduce::TestedMicrokernel;
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::reduce::{ReduceCase, ReduceMicrokernelScopeUnderTest};

/// A cube of four planes reduces every operation to the CPU's answer, rows shorter than its units
/// and several times longer, along the reduced axis and across it, each lane its own row.
#[test]
fn a_cube_reduces_every_operation_like_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases = ReduceCase::every_operation(
        TestedMicrokernel::CubeStaged { planes: 4 },
        &[(37, 1), (4096, 4)],
        &[(37, 1), (300, 4)],
    );
    ReduceMicrokernelScopeUnderTest { hw }.check(cases);
}
