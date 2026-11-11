//! The plane shuffle microkernel alone: one plane's units take `K` in turns, each summing its share in registers.

use crate::launch::matmul::{MatmulElements, PlaneShuffleLaunch, WholeRows};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::plane::{PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

/// The rows the product holds whole.
const ROWS: WholeRows = WholeRows(4);

impl MatmulSettings for PlaneShuffleMicrokernelSettings {
    type Microkernel = PlaneShuffleMicrokernel;
    type Launch = PlaneShuffleLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<PlaneShuffleLaunch, Refusal> {
        PlaneShuffleLaunch::new(hw, elements, self, ROWS, steps)
    }
}

/// Every plane shuffle microkernel the test device runs, at each of its settings, over one and
/// three of its steps of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_shuffle_matmul_matches_the_cpu() {
    let matmul = Matmul::<PlaneShuffleMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
