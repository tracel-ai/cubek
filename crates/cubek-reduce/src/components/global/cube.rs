use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::{
        global::reduce_count,
        instructions::{SharedAccumulator, fuse_accumulator_inplace, reduce_inplace},
        readers::{Reader, cube::CubeReader},
        writer::Writer,
    },
    routines::CubeBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullCubeReduce;

#[cube]
impl GlobalFullCubeReduce {
    pub fn execute<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        inst: &I,
        #[comptime] line_mode: LineMode,
        #[comptime] blueprint: CubeBlueprint,
    ) {
        let write_index = CUBE_POS;

        let input_line_size = input.line_size();
        let accumulator_size = blueprint.num_shared_accumulators;
        let worker_pos = Self::worker_pos(blueprint);

        let mut writer =
            Writer::<Out>::new::<P>(input, output, reduce_axis, write_index, line_mode);

        let write_count = writer.write_count();

        let reduce_index_start = write_index * write_count;

        if comptime![blueprint.cube_idle] {
            let reduce_count = reduce_count(
                output.len() * output.line_size(),
                line_mode,
                input.line_size(),
            );

            if reduce_index_start >= reduce_count {
                terminate!();
            }
        }

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;

            let mut accumulator_shared = Self::reduce_shared::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                line_mode,
                blueprint,
            );

            let mut accumulator_final = I::null_accumulator(inst, input_line_size);

            match comptime!(blueprint.use_planes) {
                true => {
                    if worker_pos == 0 {
                        reduce_scan::<P, I>(
                            inst,
                            &mut accumulator_shared,
                            &mut accumulator_final,
                            accumulator_size,
                        );
                        writer.write::<P, I>(b, accumulator_final, inst);
                    }
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
                        writer.write::<P, I>(b, accumulator_final, inst);
                    }
                }
            };
        }

        let commit_required = writer.commit_required();

        #[allow(clippy::collapsible_if)]
        if comptime!(commit_required) {
            if worker_pos == 0u32 {
                writer.commit();
            }
        }
    }

    fn worker_pos(#[comptime] blueprint: CubeBlueprint) -> u32 {
        match comptime!(blueprint.use_planes) {
            true => UNIT_POS_Y,
            false => UNIT_POS,
        }
    }

    fn reduce_shared<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        reduce_index: u32,
        inst: &I,
        #[comptime] line_mode: LineMode,
        #[comptime] blueprint: CubeBlueprint,
    ) -> I::SharedAccumulator {
        let input_line_size = input.line_size();

        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            blueprint.bound_checks,
            line_mode,
        );
        let reader = CubeReader::<P>::new(reader);
        let mut accumulator = I::null_accumulator(inst, input_line_size);

        for i in 0..reader.length() {
            let (item, coordinate) = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, coordinate, false);
        }

        let worker_pos = Self::worker_pos(blueprint);

        let accumulator_plane = match comptime!(blueprint.use_planes) {
            true => {
                // Sync at the plane level.
                let (item, coordinate) = I::read_accumulator(inst, &accumulator);
                let mut accumulator_plane = I::null_accumulator(inst, input_line_size);
                reduce_inplace::<P, I>(inst, &mut accumulator_plane, item, coordinate, true);
                accumulator_plane
            }
            false => accumulator,
        };

        // Sync at the cube level.
        let accumulator_size = blueprint.num_shared_accumulators;
        let requirements = I::requirements(inst);
        let mut accumulator_shared = I::SharedAccumulator::allocate(
            accumulator_size,
            input_line_size,
            requirements.coordinates,
        );

        I::SharedAccumulator::write(&mut accumulator_shared, worker_pos, accumulator_plane);

        sync_cube();

        accumulator_shared
    }
}

#[cube]
fn reduce_scan<P: ReducePrecision, I: ReduceInstruction<P>>(
    inst: &I,
    accumulator: &mut I::SharedAccumulator,
    result: &mut I::AccumulatorItem,
    #[comptime] size: u32,
) {
    for i in 0..size {
        let item = I::SharedAccumulator::read(accumulator, i);
        let (item, coordinate) = I::read_accumulator(inst, &item);
        reduce_inplace::<P, I>(inst, result, item, coordinate, false);
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
    accumulator: &mut I::SharedAccumulator,
    result: &mut I::AccumulatorItem,
    worker_index: u32,
    #[comptime] size: u32,
) {
    if comptime!(size.is_power_of_two()) {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * worker_index;
            let origin = jump * (2 * worker_index + 1);
            if worker_index < num_active_units {
                fuse_accumulator_inplace::<P, I>(inst, accumulator, destination, origin);
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
                fuse_accumulator_inplace::<P, I>(inst, accumulator, destination, origin);
            }
            num_remaining_items = num_remaining_items.div_ceil(2);
            jump *= 2;
            sync_cube();
        }
    }
    sync_cube();

    let tmp = I::SharedAccumulator::read(accumulator, 0);
    let (item, coordinate) = I::read_accumulator(inst, &tmp);
    reduce_inplace::<P, I>(inst, result, item, coordinate, false);
}
