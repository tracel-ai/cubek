use super::base::CubeDeltaWalkMicrokernel;
use crate::linear_recurrence::walk::cube::{HeldState, StateAccess, UnitPartials};
use crate::linear_recurrence::walk::{DECAY_RATE, STRENGTH_RATE};
use crate::linear_recurrence::{KEY_FEATURE, POSITION, VALUE_FEATURE};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's runs of value columns, `microkernel.shape.units()` units each, walked through every position:
/// each position decays the state, adds the delta between the value and what the state recalls of
/// the key along the key, and reads the written state with the query, the read rounded to `E`
/// where it is written. The state after each position from `state.first_kept` on is written to the
/// next of the states `state` keeps.
///
/// Every operand is windowed to the cube's box: its sequence, its head, its runs of columns and
/// every key row; the queries and keys to the key head the head is grouped under. Each unit then
/// windows them to its own key rows, and a position at a time. The cube's runs walk the positions
/// together, since their units meet in its shared memory.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_delta_walk_microkernel<E: Float, N: Size>(
    cube: &Region,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    rates: &Tile,
    state: &StateAccess,
    out: &Tile,
    #[comptime] microkernel: CubeDeltaWalkMicrokernel,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let parts = comptime!(microkernel.shape.columns() / width);
    let mut write = Array::<Vector<f32, N>>::new(parts);

    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut partials =
                UnitPartials::<N>::new(run.coord(VALUE_FEATURE), slot, runs, microkernel.shape);
            let mut held =
                HeldState::<N>::load(&state.read.at(&unit), state.fresh, microkernel.shape);

            for step in unit.walk() {
                let (keys, queries) = (key.at(&step), query.at(&step));
                let value_window = value.at(&step);
                let values = value_window.flat::<f32, N>();
                let rates_window = rates.at(&step);
                let rate = rates_window.flat::<f32, Const<1>>();
                let factor = f32::exp(rate.read(DECAY_RATE).extract(0usize));
                let strength_all =
                    Vector::<f32, N>::cast_from(rate.read(STRENGTH_RATE).extract(0usize));

                held.decay_and_recall(&keys, factor, &mut partials);
                partials.sum();

                #[unroll]
                for part in 0..parts {
                    write[part] = (values.read(part) - partials.get(part)) * strength_all;
                }

                held.write_and_answer(&keys, &queries, &write, &mut partials);
                partials.sum();

                if slot == 0 {
                    let mut out_window = out.at(&step);
                    let mut writes = out_window.flat_mut::<E, N>();
                    #[unroll]
                    for part in 0..parts {
                        writes.write(part, Vector::<E, N>::cast_from(partials.get(part)));
                    }
                }

                let at = step.coord(POSITION);
                if at >= state.first_kept {
                    held.store(&mut state.write.at(&unit), at - state.first_kept);
                }
            }
        }
    }
}
