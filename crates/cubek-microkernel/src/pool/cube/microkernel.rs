use cubecl::prelude::*;
use cubecl::tile::{Monoid, Region, Tile};
use cubecl::{self as cubecl, cube};

/// One block of the cube's walk: this plane's rows, then this unit's columns and channel lines,
/// each output cell the largest tap of its window of `input`. The reduce reads the max's identity
/// past the input's edge, so a padded tap never wins.
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
