use super::base::CubeAdditiveWalkMicrokernel;
use crate::linear_recurrence::walk::cube::{
    HeldState, Partials, RowRead, RowUpdate, StateAccess, UnitMeet,
};
use crate::linear_recurrence::walk::{DECAY_RATE, STRENGTH_RATE};
use crate::linear_recurrence::{KEY_FEATURE, POSITION, VALUE_FEATURE};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's runs of value columns, `microkernel.shape.units()` units each, walked through every
/// position: each position decays the state, adds the value, scaled by the strength, along the key,
/// and reads the written state with the query. The read, plus the value scaled by `skip`, is
/// rounded to `E` where it is written. The state after each position from `state.first_kept` on is
/// written to the next of the states `state` keeps, every operand windowed as the delta walk's are.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_additive_walk_microkernel<E: Float, N: Size>(
    cube: &Region,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    rates: &Tile,
    skip: &Tile,
    state: &StateAccess,
    out: &Tile,
    #[comptime] microkernel: CubeAdditiveWalkMicrokernel,
    #[comptime] runs: usize,
) {
    let width = N::value();
    let parts = comptime!(microkernel.shape.columns() / width);
    let mut write = Array::<Vector<f32, N>>::new(parts);
    let skip_all =
        Vector::<f32, N>::cast_from(skip.flat::<f32, Const<1>>().read(0usize).extract(0usize));

    for run in cube {
        for unit in run.walk() {
            let slot = unit.coord(KEY_FEATURE);
            let mut meet =
                UnitMeet::<N>::new(run.coord(VALUE_FEATURE), slot, runs, microkernel.shape);
            let mut partials = Partials::<N>::new(microkernel.shape);
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

                #[unroll]
                for part in 0..parts {
                    write[part] = values.read(part) * strength_all;
                }

                let at = step.coord(POSITION);
                held.pass(
                    &keys,
                    &queries,
                    factor,
                    &write,
                    &mut partials,
                    RowUpdate::DecayWrite,
                    RowRead::Queries,
                );
                meet.sum(&mut partials);

                if slot == 0 {
                    let mut out_window = out.at(&step);
                    let mut writes = out_window.flat_mut::<E, N>();
                    #[unroll]
                    for part in 0..parts {
                        let read = partials.get(part) + skip_all * values.read(part);
                        writes.write(part, Vector::<E, N>::cast_from(read));
                    }
                }

                if at >= state.first_kept {
                    held.store(&mut state.write.at(&unit), at - state.first_kept);
                }
            }
        }
    }
}
