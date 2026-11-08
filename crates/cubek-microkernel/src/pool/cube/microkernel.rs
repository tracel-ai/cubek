use super::base::CubePoolMicrokernel;
use cubecl::prelude::*;
use cubecl::tile::{Monoid, Region, Semiring, Tile};
use cubecl::{self as cubecl, cube};

/// One block of the cube's walk under a max: this plane's rows, then this unit's columns and
/// channel lines, each output cell the largest tap of its window of `input`.
#[cube]
pub fn cube_max_pool_microkernel(block: &Region, output: &Tile, input: &Tile) {
    for plane in block {
        let output = output.at(&plane);
        let input = input.at(&plane);
        for cell in plane {
            let mut max = output.at(&cell);
            max.reduce_axis(&input.at(&cell), Monoid::Max);
        }
    }
}

/// One block of the cube's walk under an average: each output cell contracts its window of
/// `input` against `weights`, in registers.
#[cube]
pub fn cube_average_pool_microkernel(
    block: &Region,
    output: &Tile,
    weights: &Tile,
    input: &Tile,
    #[comptime] microkernel: CubePoolMicrokernel,
) {
    for plane in block {
        let output = output.at(&plane);
        let weights = weights.at(&plane);
        let input = input.at(&plane);
        for cell in plane {
            let mut sum = output
                .at(&cell)
                .accumulating(microkernel.register_block, Semiring::SUM_PROD);
            sum.mm(&weights.at(&cell), &input.at(&cell));
        }
    }
}
