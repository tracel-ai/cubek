//! The unit registers microkernel alone: each unit of a plane its own block of the output.

use crate::launch::matmul::{MatmulElements, UnitRegistersLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

impl MatmulSettings for UnitRegistersMicrokernelSettings {
    type Microkernel = UnitRegistersMicrokernel;
    type Launch = UnitRegistersLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<UnitRegistersLaunch, Refusal> {
        UnitRegistersLaunch::new(hw, elements, self, steps)
    }
}

/// Every unit registers microkernel the test device runs, at each of its settings, over one and
/// three of its steps of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn unit_registers_matmul_matches_the_cpu() {
    let matmul = Matmul::<UnitRegistersMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
