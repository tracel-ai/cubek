use crate::{
    components::{
        readers::{GlobalMemoryReader, ReaderType, SharedMemoryReader},
        writers::Writer,
    },
    definition::{
        InterpolateMode, InterpolateOptions, InterpolatePrecision, Transform, compute_value,
        get_halo, tile_absolute_coords,
    },
    routines::{GlobalInterpolateBlueprint, InterpolateBlueprint},
};
use cubecl::{
    prelude::*,
    std::{FastDivmod, FastDivmodExpand},
};

#[cube]
pub fn execute_interpolate<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    output: &mut Tensor<Vector<P::EI, N>>,
    num_vectors: FastDivmod<usize>,
    cubes_per_batch: FastDivmod<usize>,
    #[comptime] blueprint: InterpolateBlueprint,
) {
    let (batch, cube_pos) = cubes_per_batch.div_mod(CUBE_POS);

    let (output_height, output_width) = (output.shape(1), output.shape(2));
    let (input_height, input_width) = (input.shape(1), input.shape(2));

    let reader = get_reader::<P, N>(
        input,
        cube_pos,
        batch,
        input_height,
        input_width,
        output_width,
        blueprint,
    );

    let vector_size = N::value();

    let tile_size_area = blueprint.tile_size.area();

    let num_vectors_value = match num_vectors {
        FastDivmod::Fast { divisor, .. } => divisor,
        FastDivmod::Fallback { divisor } => divisor,
    };

    let unit_pos = UNIT_POS as usize;
    let cube_dim = CUBE_DIM as usize;
    let num_iterations = (tile_size_area * num_vectors_value - unit_pos).div_ceil(cube_dim);

    for i in 0..num_iterations {
        let thread_pos = unit_pos + i * cube_dim;

        let (unit_pos, vector_index) = num_vectors.div_mod(thread_pos);

        let (output_row, output_col) =
            tile_absolute_coords(output_width, cube_pos, unit_pos, blueprint);

        if output_col < output_width && output_row < output_height {
            let (input_row, input_col) =
                compute_input_coords::<P::EA>(output_row, output_col, blueprint);

            let (input_row_floor, input_col_floor) = (
                get_value_floor::<P::EA>(input_row, blueprint.options),
                get_value_floor::<P::EA>(input_col, blueprint.options),
            );

            let (frac_row, frac_col) = (input_row - input_row_floor, input_col - input_col_floor);

            let final_value = compute_value::<P, N>(
                input,
                input_height,
                input_width,
                isize::cast_from(input_row_floor),
                isize::cast_from(input_col_floor),
                frac_row,
                frac_col,
                vector_index,
                &reader,
                blueprint,
            );

            Writer::write(
                output,
                batch,
                vector_index,
                output_row,
                output_col,
                vector_size,
                final_value,
            );
        }
    }
}

// Computes the input coordinates corresponding to an output coordinates.
#[cube]
fn compute_input_coords<EA: Float>(
    output_row: usize,
    output_col: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> (EA, EA) {
    (
        get_input_coord::<EA>(output_row, blueprint.transform_height),
        get_input_coord::<EA>(output_col, blueprint.transform_width),
    )
}

#[cube]
fn get_input_coord<EA: Float>(coord: usize, #[comptime] transform: Transform) -> EA {
    let scale =
        EA::cast_from(transform.scale_numerator) / EA::cast_from(transform.scale_denominator);
    let offset =
        EA::cast_from(transform.offset_numerator) / EA::cast_from(transform.offset_denominator);

    EA::cast_from(coord) * scale + offset
}

#[cube]
fn get_reader<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    cube_pos: usize,
    batch: usize,
    input_height: usize,
    input_width: usize,
    output_width: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> ReaderType<P::EI, N> {
    let vector_size = N::value();

    match blueprint.global {
        GlobalInterpolateBlueprint::GlobalMemoryBlueprint(_global_memory_blueprint) => {
            ReaderType::<P::EI, N>::new_Global(GlobalMemoryReader::new(
                input,
                batch,
                input_height,
                input_width,
                vector_size,
            ))
        }
        GlobalInterpolateBlueprint::SharedMemoryBlueprint(shared_memory_blueprint) => {
            let halo = comptime!(get_halo(blueprint.options.mode));
            let radius_offset = (halo - 1) / 2;

            let (tile_row, tile_col) = tile_absolute_coords(output_width, cube_pos, 0, blueprint);

            let (tile_mapped_row, tile_mapped_col) =
                compute_input_coords::<P::EA>(tile_row, tile_col, blueprint);

            let (tile_base_row, tile_base_col) = (
                get_value_floor::<P::EA>(tile_mapped_row, blueprint.options),
                get_value_floor::<P::EA>(tile_mapped_col, blueprint.options),
            );

            let min_row = isize::cast_from(tile_base_row) - radius_offset as isize;
            let min_col = isize::cast_from(tile_base_col) - radius_offset as isize;

            ReaderType::new_Shared(SharedMemoryReader::new(
                input,
                batch,
                input_height,
                input_width,
                min_row,
                min_col,
                vector_size,
                shared_memory_blueprint,
            ))
        }
    }
}

#[cube]
fn get_value_floor<EA: Float>(value: EA, #[comptime] options: InterpolateOptions) -> EA {
    let float_precision = EA::EPSILON;
    match options.mode {
        InterpolateMode::Nearest(_) => (value + float_precision).floor(),
        _ => value.floor(),
    }
}
