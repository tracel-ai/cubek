//! The plane window microkernel alone: the units of one plane each accumulating their block of the
//! output as a memory window, on a depth of `K` no step of a block divides.

use crate::launch::matmul::{MatmulElements, MatmulLaunch, PlaneWindowLaunch};
use crate::matmul::{F16, F32, HostMatmul};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, ScopeUnderTest, Written};

/// A plane of `units` units dealt `blocks` blocks of the output, `depth` deep, of `elements`.
#[derive(Debug, Clone, Copy)]
struct WindowCase {
    elements: MatmulElements,
    settings: UnitRegistersMicrokernelSettings,
    blocks: usize,
    units: u32,
    depth: usize,
}

struct PlaneWindow {
    hw: Hardware,
}

impl PlaneWindow {
    fn launch_of(&self, case: &WindowCase) -> Result<PlaneWindowLaunch, Refusal> {
        PlaneWindowLaunch::new(
            &self.hw,
            case.elements,
            &case.settings,
            case.blocks,
            case.units,
            case.depth,
        )
    }

    /// Every setting the device lists, for `f16` and `f32` inputs summed in `f32`, three blocks
    /// dealt to two units and one to one, over a depth of seven.
    fn every_case(&self) -> Vec<WindowCase> {
        UnitRegistersMicrokernel::settings(&self.hw)
            .into_iter()
            .flat_map(|settings| {
                [F16, F32].into_iter().flat_map(move |input| {
                    let elements = MatmulElements {
                        input,
                        sum: F32,
                        out: F32,
                    };
                    [(3, 2), (1, 1)].map(|(blocks, units)| WindowCase {
                        elements,
                        settings,
                        blocks,
                        units,
                        depth: 7,
                    })
                })
            })
            .collect()
    }
}

impl ScopeUnderTest for PlaneWindow {
    type Case = WindowCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        HostMatmul::written_by(&self.launch_of(case)?, client, 0.0)
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_of(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        vec![HostMatmul::new(launch.shape()).out()]
    }

    /// Whole numbers whose products and sums every element here holds exactly.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        1e-2
    }
}

#[test]
fn plane_window_accumulates_every_unit_block_of_a_plane() {
    let window = PlaneWindow {
        hw: Hardware::from_client(&cubecl::test_device().client()),
    };
    window.check_some(window.every_case());
}
