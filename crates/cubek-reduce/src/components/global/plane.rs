use crate::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericLine,
        global::idle_check,
        instructions::{Accumulator, reduce_inplace},
        readers::{Reader, plane::PlaneReader},
        writer::Writer,
    },
    routines::{PlaneMergeStrategy, PlaneReduceBlueprint},
};

use crate::components::instructions::ReduceStep;
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullPlaneReduce;

#[cube]
impl GlobalFullPlaneReduce {
    pub fn execute<P: ReducePrecision, Out: NumericLine, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: PlaneReduceBlueprint,
    ) {
        // TODO: need a better strategy for excess units
        // The early exit below is required for invalid units on some integrated GPUs,
        // but it's invalid non-uniform control flow on WebGPU (wasm).
        // #[allow(clippy::collapsible_if)]
        // if comptime!(blueprint.plane_dim_ceil) {
        //     if UNIT_POS_X >= PLANE_DIM {
        //         terminate!();
        //     }
        // }
        let write_index = CUBE_POS * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;

        let mut writer =
            Writer::<Out>::new::<P>(input, output, reduce_axis, write_index, vectorization_mode);

        let write_count = writer.write_count();
        let reduce_index_start = write_index * write_count;

        let idle = idle_check::<P, Out>(
            input,
            output,
            reduce_index_start,
            vectorization_mode,
            blueprint.plane_idle,
        );

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;
            let result = Self::reduce_single::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                idle,
                vectorization_mode,
                blueprint,
            );

            if UNIT_POS_X == 0 {
                writer.write::<P, I>(b, result, inst);
            }
        }

        let commit_required = writer.commit_required();

        #[allow(clippy::collapsible_if)]
        if commit_required {
            if UNIT_POS_X == 0u32 {
                writer.commit();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_single<P: ReducePrecision, Out: NumericLine, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        reduce_index: usize,
        inst: &I,
        idle: ComptimeOption<bool>,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: PlaneReduceBlueprint,
    ) -> Accumulator<P> {
        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            idle,
            blueprint.bound_checks,
            vectorization_mode,
            blueprint.plane_dim_ceil,
        );
        let reader = PlaneReader::<P>::new(reader);

        let mut accumulator = I::null_accumulator(inst);

        let iteration_plane_reduce_mode = match blueprint.plane_merge_strategy {
            PlaneMergeStrategy::Eager => ReduceStep::Plane,
            PlaneMergeStrategy::Lazy => ReduceStep::Identity,
        };
        for i in 0..reader.length() {
            let item = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, iteration_plane_reduce_mode);
        }

        match blueprint.plane_merge_strategy {
            PlaneMergeStrategy::Lazy => {
                I::plane_reduce_inplace(inst, &mut accumulator);
                accumulator
            }
            PlaneMergeStrategy::Eager => accumulator,
        }
    }
}
