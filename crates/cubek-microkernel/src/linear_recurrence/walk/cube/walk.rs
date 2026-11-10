use super::held_state::HeldState;
use super::meet::UnitMeet;
use super::partials::Partials;
use super::shape::HeldStateShape;
use super::state_access::StateAccess;
use super::step::{PositionReads, additive_step, delta_step};
use crate::linear_recurrence::walk::UpdateRule;
use crate::linear_recurrence::{KEY_FEATURE, POSITION, VALUE_FEATURE};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's runs of value columns, `shape.units()` units each, walked through every position
/// under `rule`, the read of each position rounded to `E` where it is written. Under the additive
/// rule `skip` holds the head's skip, which adds its multiple of the value to the read. The state
/// after each position from `state.first_kept` on is written to the next of the states `state`
/// keeps.
///
/// Every operand is windowed to the cube's box: its sequence, its head, its runs of columns and
/// every key row; the queries and keys to the key head the head is grouped under. Each unit then
/// windows them to its own key rows, and a position at a time. The cube's runs walk the positions
/// together, since their units meet in its shared memory.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_walk_microkernel<E: Float, N: Size>(
    cube: &Region,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    rates: &Tile,
    skip: &ComptimeOption<Tile>,
    state: &StateAccess,
    out: &Tile,
    #[comptime] shape: HeldStateShape,
    #[comptime] rule: UpdateRule,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let parts = comptime!(shape.columns() / width);
    let mut write = Array::<Vector<f32, N>>::new(parts);
    let skip_all = skip.clone().map(|skip| {
        Vector::<f32, N>::cast_from(skip.flat::<f32, Const<1>>().read(0usize).extract(0usize))
    });

    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut meet = UnitMeet::<N>::new(run.coord(VALUE_FEATURE), slot, runs, shape);
            let mut partials = Partials::<N>::new(shape);
            let mut held = HeldState::<N>::load(&state.read.at(&unit), state.fresh, shape);

            for step in unit.walk() {
                let at = PositionReads::<N>::new(&step, key, query, value, rates);

                match rule {
                    UpdateRule::Delta => {
                        delta_step::<N>(&mut held, &mut meet, &mut partials, &mut write, &at, parts)
                    }
                    UpdateRule::Additive => additive_step::<N>(
                        &mut held,
                        &mut meet,
                        &mut partials,
                        &mut write,
                        &at,
                        parts,
                    ),
                }

                if slot == 0 {
                    write_read::<E, N>(out, &step, &partials, &at, &skip_all, parts);
                }

                let position = step.coord(POSITION);
                if position >= state.first_kept {
                    held.store(&mut state.write.at(&unit), position - state.first_kept);
                }
            }
        }
    }
}

/// The position's read, the sum `partials` holds, written to `out` rounded to `E`; plus `skip`
/// times the value where the rule has one.
#[cube]
fn write_read<E: Float, N: Size>(
    out: &Tile,
    step: &Region,
    partials: &Partials<N>,
    at: &PositionReads<N>,
    skip_all: &ComptimeOption<Vector<f32, N>>,
    #[comptime] parts: usize,
) {
    let mut out_window = out.at(step);
    let mut writes = out_window.flat_mut::<E, N>();
    #[comptime]
    match skip_all {
        ComptimeOption::Some(skip_all) => {
            let values = at.value.flat::<f32, N>();
            #[unroll]
            for part in 0..parts {
                let read = partials.get(part) + *skip_all * values.read(part);
                writes.write(part, Vector::<E, N>::cast_from(read));
            }
        }
        ComptimeOption::None =>
        {
            #[unroll]
            for part in 0..parts {
                writes.write(part, Vector::<E, N>::cast_from(partials.get(part)));
            }
        }
    }
}
