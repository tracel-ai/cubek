use crate::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::{idle_check, reduction_output_base},
        instructions::{
            Accumulator, ReduceStep, ReduceWithIndices, SharedAccumulator,
            fuse_accumulator_inplace, reduce_inplace,
        },
        readers::{Reader, cube::CubeReader},
        writers::{IndicesWriter, ReduceWriter, Writer},
    },
    routines::CubeBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullCubeReduce;

#[cube]
impl GlobalFullCubeReduce {
    pub fn execute<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: CubeBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let write_index = reduction_output_base::<Out::T, Out::N>(
            CUBE_POS,
            &*output,
            reduce_axis,
            comptime!(acc_format.len()),
        );

        let mut out = output.clone();
        let mut writer = Writer::<Out>::new::<P>(
            input,
            &mut out,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        Self::reduce_to_writer::<P, Out, I, Writer<Out>>(
            input,
            output,
            reduce_axis,
            write_index,
            inst,
            &mut writer,
            vectorization_mode,
            blueprint,
        );
    }

    /// Same reduction as [`Self::execute`], but writing the values and their
    /// indices to two outputs from a single pass. `indices` must have the same
    /// shape and the same reduce/vec axes as `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_indices<
        P: ReducePrecision,
        Out: NumericVector,
        Idx: NumericVector,
        I: ReduceWithIndices<P>,
    >(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        indices: &mut VirtualTensor<Idx::T, Idx::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: CubeBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let write_index = reduction_output_base::<Out::T, Out::N>(
            CUBE_POS,
            &*output,
            reduce_axis,
            comptime!(acc_format.len()),
        );

        let mut out = output.clone();
        let mut idx = indices.clone();
        let mut writer = IndicesWriter::<Out, Idx>::new::<P>(
            input,
            &mut out,
            &mut idx,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        Self::reduce_to_writer::<P, Out, I, IndicesWriter<Out, Idx>>(
            input,
            output,
            reduce_axis,
            write_index,
            inst,
            &mut writer,
            vectorization_mode,
            blueprint,
        );
    }

    /// The reduction body shared by [`Self::execute`] and
    /// [`Self::execute_with_indices`], generic over how results are written.
    #[allow(clippy::too_many_arguments)]
    fn reduce_to_writer<
        P: ReducePrecision,
        Out: NumericVector,
        I: ReduceInstruction<P>,
        W: ReduceWriter<P, I>,
    >(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        write_index: usize,
        inst: &I,
        writer: &mut W,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: CubeBlueprint,
    ) {
        let accumulator_size = blueprint.num_shared_accumulators;
        let worker_pos = Self::worker_pos(blueprint);

        let write_count = W::write_count(&*writer);

        let reduce_index_start = write_index * write_count;

        let idle = idle_check::<P, Out>(
            input,
            &*output,
            reduce_index_start,
            vectorization_mode,
            blueprint.cube_idle,
        );

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;

            let mut accumulator_shared = Self::reduce_shared::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                idle,
                vectorization_mode,
                blueprint,
            );

            let mut accumulator_final = I::null_accumulator(inst);

            match blueprint.use_planes {
                true => {
                    if worker_pos == 0 {
                        reduce_scan::<P, I>(
                            inst,
                            &mut accumulator_shared,
                            &mut accumulator_final,
                            accumulator_size,
                        );
                        W::write(writer, b, accumulator_final, inst);
                    }

                    // Wait for plane 0 to finish reading SM before next iter overwrites it.
                    sync_cube();
                }
                false => {
                    reduce_tree::<P, I>(
                        inst,
                        &mut accumulator_shared,
                        &mut accumulator_final,
                        worker_pos,
                        accumulator_size,
                    );
                    if worker_pos == 0 {
                        W::write(writer, b, accumulator_final, inst);
                    }
                }
            };
        }

        let commit_required = W::commit_required(&*writer);

        #[allow(clippy::collapsible_if)]
        if commit_required {
            if worker_pos == 0 {
                W::commit(writer);
            }
        }
    }

    fn worker_pos(#[comptime] blueprint: CubeBlueprint) -> usize {
        match blueprint.use_planes {
            true => UNIT_POS_Y as usize,
            false => UNIT_POS as usize,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_shared<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        reduce_index: usize,
        inst: &I,
        idle: ComptimeOption<bool>,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: CubeBlueprint,
    ) -> I::SharedAccumulator {
        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            idle,
            blueprint.bound_checks,
            vectorization_mode,
            false,
        );
        let reader = CubeReader::<P>::new(reader);
        let mut accumulator = I::null_accumulator(inst);

        for i in 0..reader.length() {
            let item = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, ReduceStep::Identity);
        }

        let worker_pos = Self::worker_pos(blueprint);

        let accumulator_plane = match blueprint.use_planes {
            true => {
                I::plane_reduce_inplace(inst, &mut accumulator);
                accumulator
            }
            false => accumulator,
        };

        // Sync at the cube level.
        let accumulator_size = blueprint.num_shared_accumulators;
        let requirements = I::requirements(inst);
        let mut accumulator_shared =
            I::SharedAccumulator::allocate(accumulator_size, requirements.coordinates, inst);

        I::SharedAccumulator::write(&mut accumulator_shared, worker_pos, accumulator_plane);

        sync_cube();

        accumulator_shared
    }
}

#[cube]
fn reduce_scan<P: ReducePrecision, I: ReduceInstruction<P>>(
    inst: &I,
    shared_accumulator: &mut I::SharedAccumulator,
    accumulator: &mut Accumulator<P>,
    #[comptime] size: usize,
) {
    for i in 0..size {
        let acc = I::SharedAccumulator::read(&*shared_accumulator, i);
        I::fuse_accumulators(inst, accumulator, &acc);
    }
}

/// Use all units within a cube to fuse the first `size` elements of `accumulator` inplace like this with some padding if `size` is not a power of 2.
///
///
/// ```ignored
///
///     0   1   2   3   4   5   6   7
///     |   |   |   |   |   |   |   |
///     +---+   +---+   +---+   +---+
///     |       |       |       |
///     +-------+       +-------+
///     |               |
///     +---------------+
///     |
///     *
///
/// ```
///
/// The outcome is stored in the first element of the accumulator and also returned by this function for convenience.
///
/// Since each individual cube performs a reduction, this function is meant to be called
/// with a different `accumulator` for each cube based on `CUBE_POS`.
///
/// There is no out-of-bound check, so it is the responsibility of the caller to ensure that `size` is at most the length
/// of the shared memory and that there are at least `size` units within each cube.
#[cube]
fn reduce_tree<P: ReducePrecision, I: ReduceInstruction<P>>(
    inst: &I,
    shared_accumulator: &mut I::SharedAccumulator,
    accumulator: &mut Accumulator<P>,
    worker_index: usize,
    #[comptime] size: usize,
) {
    if size.is_power_of_two() {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * worker_index;
            let origin = jump * (2 * worker_index + 1);
            if worker_index < num_active_units {
                fuse_accumulator_inplace::<P, I>(inst, shared_accumulator, destination, origin);
            }
            jump *= 2;
            sync_cube();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * worker_index;
            let origin = jump * (2 * worker_index + 1);
            if worker_index < num_remaining_items / 2 {
                fuse_accumulator_inplace::<P, I>(inst, shared_accumulator, destination, origin);
            }
            num_remaining_items = num_remaining_items.div_ceil(2);
            jump *= 2;
            sync_cube();
        }
    }
    sync_cube();

    let acc = I::SharedAccumulator::read(&*shared_accumulator, 0);
    I::fuse_accumulators(inst, accumulator, &acc);
}
