use super::base::CubeRmsNormMicrokernel;
use crate::rms_norm::{HeldRow, RmsNormScale};
use crate::row_store::{CubeStage, CubeStore};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Monoid, Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's row: each of `cube`'s planes, a run of the row each, sums its run across its
/// lanes, its first lane leaves the sum in the plane's slot, and once every plane has, each unit
/// adds the slots and writes its lines of the run scaled. The cube meets again once every unit has
/// read the slots, so a cube walking a next row cannot overwrite them first. `input` and `weight` are windowed to the
/// cube's row, and `out` is the cube's window of the output, which its planes write their runs
/// through, straight or staged and encoded as `stage` says.
#[cube]
pub fn cube_rms_norm_microkernel<E: Float, N: Size>(
    cube: &Region,
    input: &Tile,
    weight: &Tile,
    out: &Tile,
    scale: &RmsNormScale,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[comptime] stage: CubeStage,
) {
    let mut store = CubeStore::new::<E, N>(out.clone(), stage);
    let planes = comptime!(microkernel.planes);
    for plane in cube {
        // The cube's planes lie along `y`, as its resources state them.
        let (slot, lane) = (UNIT_POS_Y as usize, UNIT_POS_PLANE as usize);
        let mut row = HeldRow::<E, N>::new(comptime!(microkernel.plane.held_row()));
        let run = UnitShare::Plane.reduce::<f32>(row.load(&input.at(&plane), lane), Monoid::Sum);
        let mut slots = Shared::<[f32]>::new_slice(planes);
        if lane == 0 {
            slots[slot] = run;
        }
        sync_cube();
        let mut sum = 0.0f32;
        #[unroll]
        for p in 0..planes {
            sum += slots[p];
        }
        sync_cube();
        let run_lines = comptime!(microkernel.plane.held_row().lines);
        let mut run_out = store.owner(&plane, slot * run_lines);
        row.write(&weight.at(&plane), &mut run_out, lane, scale.of(sum));
    }
    store.finish();
}
