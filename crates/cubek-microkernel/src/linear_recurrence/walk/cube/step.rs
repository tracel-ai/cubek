use super::held_state::HeldState;
use super::meet::UnitMeet;
use super::partials::Partials;
use super::row_pass::{RowRead, RowUpdate};
use crate::linear_recurrence::walk::{DECAY_RATE, STRENGTH_RATE};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// What one position reads: the windows of its key, query and value, and its two rates.
#[derive(CubeType)]
pub struct PositionReads<N: Size> {
    pub keys: Tile,
    pub queries: Tile,
    pub value: Tile,
    pub factor: f32,
    pub strength: Vector<f32, N>,
}

#[cube]
impl<N: Size> PositionReads<N> {
    /// The windows of `step` of the unit's key, query, value and rates, and the rates read: the
    /// factor `exp(decay)` and the strength.
    pub fn new(step: &Region, key: &Tile, query: &Tile, value: &Tile, rates: &Tile) -> Self {
        let rates_window = rates.at(step);
        let rate = rates_window.flat::<f32, Const<1>>();
        let factor = f32::exp(rate.read(DECAY_RATE).extract(0usize));
        let strength = Vector::<f32, N>::cast_from(rate.read(STRENGTH_RATE).extract(0usize));
        PositionReads::<N> {
            keys: key.at(step),
            queries: query.at(step),
            value: value.at(step),
            factor,
            strength,
        }
    }
}

/// One position of the delta rule: the state decayed and what it recalls of the key met, the
/// delta between the value and that recall written along the key, and what the written state
/// answers to the query met, left in `partials`.
#[cube]
pub fn delta_step<N: Size>(
    held: &mut HeldState<N>,
    meet: &mut UnitMeet<N>,
    partials: &mut Partials<N>,
    write: &mut Array<Vector<f32, N>>,
    at: &PositionReads<N>,
    #[comptime] parts: usize,
) {
    let values = at.value.flat::<f32, N>();
    held.pass(
        &at.keys,
        &at.queries,
        at.factor,
        &*write,
        partials,
        RowUpdate::Decay,
        RowRead::Keys,
    );
    meet.sum(partials);

    #[unroll]
    for part in 0..parts {
        write[part] = (values.read(part) - partials.get(part)) * at.strength;
    }

    held.pass(
        &at.keys,
        &at.queries,
        at.factor,
        &*write,
        partials,
        RowUpdate::Write,
        RowRead::Queries,
    );
    meet.sum(partials);
}

/// One position of the additive rule: the value, scaled by the strength, written along the key of
/// the decayed state, and what that answers to the query met, left in `partials`.
#[cube]
pub fn additive_step<N: Size>(
    held: &mut HeldState<N>,
    meet: &mut UnitMeet<N>,
    partials: &mut Partials<N>,
    write: &mut Array<Vector<f32, N>>,
    at: &PositionReads<N>,
    #[comptime] parts: usize,
) {
    let values = at.value.flat::<f32, N>();
    #[unroll]
    for part in 0..parts {
        write[part] = values.read(part) * at.strength;
    }

    held.pass(
        &at.keys,
        &at.queries,
        at.factor,
        &*write,
        partials,
        RowUpdate::DecayWrite,
        RowRead::Queries,
    );
    meet.sum(partials);
}
