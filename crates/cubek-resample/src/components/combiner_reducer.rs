use crate::definition::{Resample, Semiring};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};
use std::hash::Hash;

#[cube]
pub trait CombinerReducer<F: Float, N: Size>: Send + Sync + 'static {
    type Accumulator: CubeType + CubePrimitive;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;
    type Indices: LaunchArg;

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator;

    fn combine(
        value: &mut Vector<F, N>,
        weight: Vector<F, N>,
        tap_idx: usize,
        #[comptime] config: &Self::Config,
    );

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        combined: Vector<F, N>,
        tap_idx: usize,
        #[comptime] config: &Self::Config,
    );

    fn count_position(
        accumulator: &mut Self::Accumulator,
        position: &CoordsDyn,
        #[comptime] config: &Self::Config,
    );

    fn store(
        position: CoordsDyn,
        output: &mut ViewMut<Vector<F, N>, CoordsDyn>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
        #[comptime] config: &Self::Config,
    );
}

pub struct AccumulateCombinerReducer;

#[cube]
impl<F: Float, N: Size> CombinerReducer<F, N> for AccumulateCombinerReducer {
    type Accumulator = Vector<F, N>;
    type Config = Resample;
    type Indices = ();

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator {
        Semiring::identity(&config.semiring)
    }

    fn combine(
        value: &mut Vector<F, N>,
        weight: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Self::Config,
    ) {
        *value = Semiring::combine(*value, weight, &config.semiring)
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        combined: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Self::Config,
    ) {
        *accumulator = Semiring::reduce(*accumulator, combined, &config.semiring);
    }

    fn count_position(
        _accumulator: &mut Self::Accumulator,
        _position: &CoordsDyn,
        #[comptime] _config: &Self::Config,
    ) {
    }

    fn store(
        position: CoordsDyn,
        output: &mut ViewMut<Vector<F, N>, CoordsDyn>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
        #[comptime] _config: &Self::Config,
    ) {
        output.write(position, accumulator);
    }
}
