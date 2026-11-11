//! The hypercube staged microkernel alone: cubes sharing one box's `K`, each walking its run of
//! the stages, the box's last arrival adding their parts up.

use crate::launch::matmul::{HypercubeStagedLaunch, MatmulElements};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulCase, MatmulSettings};

impl MatmulSettings for HypercubeStagedMicrokernelSettings {
    type Microkernel = HypercubeStagedMicrokernel;
    type Launch = HypercubeStagedLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<HypercubeStagedLaunch, Refusal> {
        HypercubeStagedLaunch::new(hw, elements, self, steps)
    }
}

/// Every hypercube staged microkernel the test device runs, at each of its settings, over two
/// boxes whose stages four cubes share: one stage deep, where two cubes hold a box whole and two
/// walk nothing; three, where runs cross from one box into the next; and five, where runs of three
/// cut both boxes unevenly; for `f16` and `f32` inputs summed in `f32`.
#[test]
fn hypercube_staged_matmul_matches_the_cpu() {
    let matmul = Matmul::<HypercubeStagedMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    let cases: Vec<_> = matmul
        .every_setting()
        .into_iter()
        .filter(|case| case.steps == 1)
        .flat_map(|case| [1, 3, 5].map(|steps| MatmulCase { steps, ..case }))
        .collect();
    matmul.check(cases);
}
