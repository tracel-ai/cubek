use super::base::CubeGroupsRmsNormMicrokernel;
use crate::rms_norm::{HeldRow, RmsNormScale};
use crate::row_store::{CubeStage, CubeStore};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, a group of `microkernel.units` units each: `out = input · scale · weight`,
/// each unit of a group its lines of the group's row. `input` and `weight` are windowed to the
/// cube's rows, `groups` of them, and `out` is the cube's window of the output, which its groups
/// write through one store, straight or staged and encoded as `stage` says.
///
/// A group of one sums nothing. A larger group's units leave their partials in one slot each of
/// the group's run of the cube's shared memory, and sum them in halves, the cube meeting between
/// every step: its groups step together, each in its own run.
#[cube]
pub fn cube_groups_rms_norm_microkernel<E: Float, N: Size>(
    cube: &Region,
    input: &Tile,
    weight: &Tile,
    out: &Tile,
    scale: &RmsNormScale,
    #[comptime] microkernel: CubeGroupsRmsNormMicrokernel,
    #[comptime] groups: usize,
    #[comptime] stage: CubeStage,
) {
    let mut store = CubeStore::new::<E, N>(out.clone(), stage);
    let units = comptime!(microkernel.units);
    for group in cube {
        // A group's units lie along `x` and the cube's groups along `y`, as its resources state
        // them.
        let (slot, unit) = (UNIT_POS_Y as usize, UNIT_POS_X as usize);
        let mut row = HeldRow::<E, N>::new(comptime!(microkernel.held_row()));
        let partial = row.load(&input.at(&group), unit);
        let mut sum = partial;
        if comptime!(units > 1) {
            let mut slots = Shared::<[f32]>::new_slice(comptime!(groups * units));
            let first = slot * units;
            slots[first + unit] = partial;
            sync_cube();
            #[unroll]
            for step in 1..comptime!(units.ilog2() as usize + 1) {
                let half = comptime!(units >> step);
                if unit < half {
                    slots[first + unit] += slots[first + unit + half];
                }
                sync_cube();
            }
            sum = slots[first];
        }
        let row_lines = comptime!(microkernel.held_row().lines);
        let mut row_out = store.owner(&group, slot * row_lines);
        row.write(&weight.at(&group), &mut row_out, unit, scale.of(sum));
    }
    store.finish();
}
