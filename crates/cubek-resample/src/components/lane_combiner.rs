use crate::{
    components::semiring::{semiring_combine, semiring_combine_vec, semiring_identity},
    definition::Resample,
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[cube]
pub trait LaneCombiner<F: Float, N: Size> {
    /// Initialize the accumulator for a single tap.
    fn init_combined(#[comptime] config: &Resample) -> Vector<F, N>;

    /// Combine the value from a specific lane and update the combined vector.
    fn combine_lane(
        in_coord: &CoordsDyn,
        combined: Vector<F, N>,
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
    fn init_combined(#[comptime] _config: &Resample) -> Vector<F, N> {
        // Gets optimized by the compiler.
        Vector::new(F::new(0.0))
    }

    fn combine_lane(
        _in_coord: &CoordsDyn,
        _combined: Vector<F, N>,
        _lane: usize,
        value: Vector<F, N>,
        weight: F,
        #[comptime] config: &Resample,
        #[comptime] _vectorized_axis: usize,
        #[comptime] _vector_size: usize,
    ) -> Vector<F, N> {
        semiring_combine_vec(&config.semiring, value, Vector::new(weight))
    }
}

#[derive(CubeType)]
pub struct VectorizedLaneCombiner {}

#[cube]
impl<F: Float, N: Size> LaneCombiner<F, N> for VectorizedLaneCombiner {
    fn init_combined(#[comptime] config: &Resample) -> Vector<F, N> {
        semiring_identity::<F, N>(&config.semiring)
    }

    fn combine_lane(
        in_coord: &CoordsDyn,
        mut combined: Vector<F, N>,
        lane: usize,
        value: Vector<F, N>,
        weight: F,
        #[comptime] config: &Resample,
        #[comptime] vectorized_axis: usize,
        #[comptime] vector_size: usize,
    ) -> Vector<F, N> {
        let extract_idx = in_coord[vectorized_axis] as usize % vector_size;

        let value_lane = value.extract(extract_idx);
        let combined_lane = semiring_combine::<F>(&config.semiring, value_lane, weight);

        combined.insert(lane, combined_lane);

        combined
    }
}
