use super::base::CubeGatedDeltaWalkMicrokernel;
use super::held_state::HeldState;
use super::unit_partials::UnitPartials;
use crate::gated_delta::{KEY_FEATURE, POSITION, VALUE_FEATURE};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's runs of value columns, `microkernel.units` units each, walked through every position:
/// each position decays the state, adds the delta between the value and what the state recalls of
/// the key along the key, and reads the written state with the query, the read rounded to `E`
/// where it is written. The state after each position from `first_kept` on is written to the next
/// of the kept states.
///
/// Every operand is windowed to the cube's box: its sequence, its head, its runs of columns and
/// every key row; the queries and keys to the key head the head is grouped under. Each unit then
/// windows them to its own key rows, and a position at a time. The cube's runs walk the positions
/// together, since their units meet in its shared memory.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_gated_delta_walk_microkernel<E: Float, N: Size>(
    cube: &Region,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    decay: &Tile,
    strength: &Tile,
    state: &Tile,
    out: &Tile,
    kept: &Tile,
    first_kept: usize,
    #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let parts = comptime!(microkernel.columns / width);
    let mut write = Array::<Vector<f32, N>>::new(parts);
    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut partials =
                UnitPartials::<N>::new(run.coord(VALUE_FEATURE), slot, runs, microkernel);
            let mut held = HeldState::<N>::load(&state.at(&unit), microkernel);
            for step in unit.walk() {
                let (keys, queries) = (key.at(&step), query.at(&step));
                let value_window = value.at(&step);
                let values = value_window.flat::<f32, N>();
                let factor = f32::exp(scalar(&decay.at(&step)));
                let strength_all = Vector::<f32, N>::cast_from(scalar(&strength.at(&step)));

                held.decay_and_recall(&keys, factor, &mut partials);
                partials.sum(0usize);
                #[unroll]
                for part in 0..parts {
                    write[part] = (values.read(part) - partials.get(part)) * strength_all;
                }
                held.write_and_answer(&keys, &queries, &write, &mut partials);
                partials.sum(1usize);
                if slot == 0 {
                    let mut out_window = out.at(&step);
                    let mut writes = out_window.flat_mut::<E, N>();
                    #[unroll]
                    for part in 0..parts {
                        writes.write(part, Vector::<E, N>::cast_from(partials.get(part)));
                    }
                }
                let at = step.coord(POSITION);
                if at >= first_kept {
                    held.store(&mut kept.at(&unit), at - first_kept);
                }
            }
        }
    }
}

/// The one value a position's window of a per-head scalar holds: its decay, or its strength.
#[cube]
fn scalar(window: &Tile) -> f32 {
    window.flat::<f32, Const<1>>().read(0usize).extract(0usize)
}
