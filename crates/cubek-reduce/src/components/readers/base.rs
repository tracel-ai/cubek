use crate::{
    BoundChecks, ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericLine,
        instructions::{AccumulatorKind, ReduceCoordinate, ReduceRequirements},
        readers::{parallel::ParallelReader, perpendicular::PerpendicularReader},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub enum Reader<P: ReducePrecision> {
    Parallel(ParallelReader<P>),
    Perpendicular(PerpendicularReader<P>),
}

#[cube]
impl<P: ReducePrecision> Reader<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<I: ReduceInstruction<P>, Out: NumericLine>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        inst: &I,
        reduce_axis: usize,
        reduce_index: usize,
        idle: ComptimeOption<bool>,
        #[comptime] bound_checks: BoundChecks,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] plane_dim_ceil: bool,
    ) -> Reader<P> {
        let effective_plane_dim = if plane_dim_ceil {
            min(CUBE_DIM_X, PLANE_DIM)
        } else {
            CUBE_DIM_X
        };
        match vectorization_mode {
            VectorizationMode::Parallel => {
                Reader::<P>::new_Parallel(ParallelReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    idle,
                    effective_plane_dim,
                    bound_checks,
                ))
            }
            VectorizationMode::Perpendicular => {
                Reader::<P>::new_Perpendicular(PerpendicularReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    idle,
                    effective_plane_dim,
                    bound_checks,
                ))
            }
        }
    }
}

#[cube]
impl<N: Size> ReduceCoordinate<N> {
    pub fn new(
        coordinate: usize,
        requirements: ReduceRequirements,
        #[comptime] vectorization_mode: VectorizationMode,
    ) -> Self {
        if requirements.coordinates.comptime() {
            // TODO: Make this generic to allow 64-bit coordinate output.
            // Can't directly use `usize` for the buffer, since its size isn't defined beyond the
            // kernel boundary.
            ReduceCoordinate::new_Required(AccumulatorKind::new_single(fill_coordinate_vector(
                coordinate as u32,
                vectorization_mode,
            )))
        } else {
            ReduceCoordinate::new_NotRequired()
        }
    }
}

// If vectorization mode is parallel, fill a vector with `x, x+1, ... x+ vector_size - 1` where `x = first`.
// If vectorization mode is perpendicular, fill a vector with `x, x, ... x` where `x = first`.
#[cube]
pub(crate) fn fill_coordinate_vector<N: Size>(
    first: u32,
    #[comptime] vectorization_mode: VectorizationMode,
) -> Vector<u32, N> {
    match vectorization_mode {
        VectorizationMode::Parallel => {
            let mut coordinates = Vector::empty();
            #[unroll]
            for j in 0..N::value() {
                coordinates[j] = first + j as u32;
            }
            coordinates
        }
        VectorizationMode::Perpendicular => Vector::empty().fill(first),
    }
}
