//! The cube SwiGLU microkernel alone: each cube's group of units shares one row's run.

use cubek_microkernel::swiglu::cube::CubeSwigluMicrokernel;
use cubek_microkernel::testing::RowBatch;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::swiglu::{F16, F32, Swiglu, SwigluCase};

/// A cube gates every row as the host does, at each of its settings, for `f16` and `f32` rows:
/// one row and several, a width of whole lines, one its run divides unevenly, and one shorter
/// than a group's units.
#[test]
fn a_cube_gates_every_row_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let cases: Vec<_> = CubeSwigluMicrokernel::settings(&hw)
        .into_iter()
        .flat_map(|settings| {
            [F16, F32].into_iter().flat_map(move |element| {
                [(1, 1024), (3, 1000), (2, 70)].map(|(rows, dim)| SwigluCase {
                    element,
                    batch: RowBatch { rows, dim },
                    settings,
                })
            })
        })
        .collect();
    Swiglu { hw }.check(cases);
}
