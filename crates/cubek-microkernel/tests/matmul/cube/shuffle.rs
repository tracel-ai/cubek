//! The cube shuffle microkernel alone: a cube's planes take `K` in turns and meet in one accumulator.

use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::cube::{CubeShuffleMicrokernel, CubeShuffleMicrokernelSettings};
use cubek_microkernel::testing::matmul::{CubeShuffleLaunch, MatmulElements};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

/// The rows the product holds whole: a decode step's few.
const ROWS: usize = 4;

impl MatmulSettings for CubeShuffleMicrokernelSettings {
    type Microkernel = CubeShuffleMicrokernel;
    type Launch = CubeShuffleLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<CubeShuffleLaunch, Refusal> {
        CubeShuffleLaunch::new(hw, elements, self, ROWS, steps)
    }
}

/// Every cube shuffle microkernel the test device runs, at each of its settings, over one and three
/// of its steps of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn cube_shuffle_matmul_matches_the_cpu() {
    let matmul = Matmul::<CubeShuffleMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
