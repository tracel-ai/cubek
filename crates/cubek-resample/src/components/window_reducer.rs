use crate::{
    components::semiring::{semiring_identity, semiring_reduce},
    definition::Resample,
};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};
use std::hash::Hash;

#[cube]
pub trait WindowReducer<F: Float, N: Size>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    type Indices: LaunchArg;

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator;

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Vector<F, N>,
    );

    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        position: CoordsDyn,
    );

    fn store(
        #[comptime] config: &Self::Config,
        position: CoordsDyn,
        output: &mut ViewMut<Vector<F, N>, CoordsDyn>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
    );
}

pub struct AccumulateWindowReducer;

#[cube]
impl<F: Float, N: Size> WindowReducer<F, N> for AccumulateWindowReducer {
    type Accumulator = Vector<F, N>;
    type Config = Resample;
    type Indices = ();

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator {
        semiring_identity(&config.semiring)
    }

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: VectorSize,
        combined: Vector<F, N>,
    ) {
        *accumulator = semiring_reduce(&config.semiring, *accumulator, combined);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _position: CoordsDyn,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: CoordsDyn,
        output: &mut ViewMut<Vector<F, N>, CoordsDyn>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output.write(position, accumulator);
    }
}
