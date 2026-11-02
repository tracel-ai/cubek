use super::base::{CubeGroupsRmsNormMicrokernel, GroupSum};
use crate::rms_norm::{HeldRow, RmsNormScale};
use crate::row_store::{CubeStage, CubeStore};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Monoid, Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, a group of `microkernel.units` units each: `out = input · scale · weight`,
/// each unit of a group its lines of the group's row. `input` and `weight` are windowed to the
/// cube's rows, `groups` of them, and `out` is the cube's window of the output, which its groups
/// write through one store, straight or staged and encoded as `stage` says.
///
/// A group sums its units' partials as `microkernel.sum` says ([`GroupSum`]). Where it sums through
/// shared memory, each group has its own run of slots, and the cube meets again once every unit
/// has read the sum, so a cube walking next rows cannot overwrite the slots first.
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
        let sum = match comptime!(microkernel.sum) {
            GroupSum::Alone => partial,
            GroupSum::Shuffles => {
                let lanes = comptime!(UnitShare::Group {
                    unit_bits: units - 1
                });
                lanes.reduce::<f32>(partial, Monoid::Sum)
            }
            GroupSum::Planes { lanes } => {
                let planes = comptime!(units / lanes);
                let mut slots = Shared::<[f32]>::new_slice(comptime!(groups * planes));
                let first = slot * planes;
                let plane_sum = UnitShare::Plane.reduce::<f32>(partial, Monoid::Sum);
                if unit % lanes == 0 {
                    slots[first + unit / lanes] = plane_sum;
                }
                sync_cube();
                let mut sum = 0.0f32;
                #[unroll]
                for p in 0..planes {
                    sum += slots[first + p];
                }
                sync_cube();
                sum
            }
            GroupSum::Halving => {
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
                let sum = slots[first];
                sync_cube();
                sum
            }
        };
        let row_lines = comptime!(microkernel.held_row().lines);
        let mut row_out = store.owner(&group, slot * row_lines);
        row.write(&weight.at(&group), &mut row_out, unit, scale.of(sum));
    }
    store.finish();
}
