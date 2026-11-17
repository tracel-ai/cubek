use super::base::CubeInterpolateMicrokernel;
use cubecl::prelude::*;
use cubecl::tile::{Region, Semiring, Tile};
use cubecl::{self as cubecl, cube};

/// One block of the cube's walk: this plane's rows, then this unit's columns and channel lines,
/// each output cell contracting its whole tap window, `weights` by `input`, into `output`.
#[cube]
pub fn cube_interpolate_microkernel(
    block: &Region,
    output: &Tile,
    weights: &Tile,
    input: &Tile,
    #[comptime] microkernel: CubeInterpolateMicrokernel,
) {
    for plane in block {
        let output = output.at(&plane);
        let weights = weights.at(&plane);
        let input = input.at(&plane);
        for cell in plane {
            let mut sum = output
                .at(&cell)
                .accumulating(microkernel.software_matmul, Semiring::SUM_PROD);
            sum.mm(&weights.at(&cell), &input.at(&cell));
        }
    }
}
