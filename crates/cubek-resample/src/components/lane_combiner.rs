use crate::{
    components::semiring::{semiring_combine, semiring_combine_vec, semiring_identity},
    definition::Resample,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};
use std::hash::Hash;

#[cube]
pub trait LaneCombiner<F: Float, N: Size> {
    type Accumulator: CubeType;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    fn initialize(#[comptime] config: &Resample) -> Vector<F, N>;

    fn combine_lane(
        in_coord: &CoordsDyn,
        accumulator: Vector<F, N>,
        lane: usize,
        value: Vector<F, N>,
        weight: F,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
    ) -> Vector<F, N>;
}

#[derive(CubeType)]
pub struct ScalarLaneCombiner {}

#[cube]
impl<F: Float, N: Size> LaneCombiner<F, N> for ScalarLaneCombiner {
    type Accumulator = Vector<F, N>;
    type Config = Resample;

    fn initialize(#[comptime] _config: &Resample) -> Self::Accumulator {
        // Gets optimized by the compiler.
        Vector::new(F::new(0.0))
    }

    fn combine_lane(
        _in_coord: &CoordsDyn,
        _accumulator: Self::Accumulator,
        _lane: usize,
        value: Self::Accumulator,
        weight: F,
        #[comptime] config: &Resample,
        #[comptime] _vectorized_axis: usize,
        #[comptime] _vector_size: usize,
    ) -> Self::Accumulator {
        semiring_combine_vec(&config.semiring, value, Vector::new(weight))
    }
}

#[derive(CubeType)]
pub struct VectorizedLaneCombiner {}

#[cube]
impl<F: Float, N: Size> LaneCombiner<F, N> for VectorizedLaneCombiner {
    type Accumulator = Vector<F, N>;
    type Config = Resample;

    fn initialize(#[comptime] config: &Resample) -> Self::Accumulator {
        semiring_identity::<F, N>(&config.semiring)
    }

    fn combine_lane(
        in_coord: &CoordsDyn,
        mut accumulator: Self::Accumulator,
        lane: usize,
        value: Self::Accumulator,
        weight: F,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
    ) -> Self::Accumulator {
        let extract_idx = in_coord[vectorized_axis] as usize % vector_size;

        let value_lane = value.extract(extract_idx);
        let combined_lane = semiring_combine::<F>(&config.semiring, value_lane, weight);

        accumulator.insert(lane, combined_lane);

        accumulator
    }
}
