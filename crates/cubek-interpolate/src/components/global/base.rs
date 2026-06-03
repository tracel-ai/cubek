use crate::{
    components::{
        readers::{GlobalMemoryReader, ReaderType, SharedMemoryReader},
        writers::Writer,
    },
    definition::{
        InterpolateMode, InterpolateOptions, InterpolatePrecision, NearestMode, compute_value,
        get_halo, tile_absolute_coords,
    },
    routines::{GlobalInterpolateBlueprint, InterpolateBlueprint},
};
use cubecl::{prelude::*, std::FastDivmod};

#[cube]
pub fn execute_interpolate<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    output: &mut Tensor<Vector<P::EI, N>>,
    num_vectors: FastDivmod<usize>,
    cubes_per_batch: FastDivmod<usize>,
    #[comptime] blueprint: InterpolateBlueprint,
) {
    let (unit_pos, vector_index) = num_vectors.div_mod(UNIT_POS as usize);
    let (batch, cube_pos) = cubes_per_batch.div_mod(CUBE_POS);

    let (output_height, output_width) = (output.shape(1), output.shape(2));
    let (input_height, input_width) = (input.shape(1), input.shape(2));

    let (output_row, output_col) = tile_absolute_coords(
        output_width,
        cube_pos,
        unit_pos,
        blueprint.tile_size,
        blueprint.options,
    );

    let (input_row, input_col) = compute_input_coords::<P::EA>(
        output_row,
        output_col,
        input_height,
        input_width,
        output_height,
        output_width,
        blueprint.options,
    );

    let (input_row_floor, input_col_floor) = (
        get_value_floor::<P::EA>(input_row, blueprint.options),
        get_value_floor::<P::EA>(input_col, blueprint.options),
    );

    let (frac_row, frac_col) = (input_row - input_row_floor, input_col - input_col_floor);

    let vector_size = N::value();

    let reader = get_reader::<P, N>(
        input,
        cube_pos,
        batch,
        vector_index,
        input_height,
        input_width,
        output_height,
        output_width,
        blueprint,
    );

    let final_value = compute_value::<P, N>(
        input,
        input_height,
        input_width,
        isize::cast_from(input_row_floor),
        isize::cast_from(input_col_floor),
        frac_row,
        frac_col,
        reader,
        blueprint,
    );

    if output_col < output_width && output_row < output_height {
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

// Computes the input coordinates corresponding to an output coordinates.
#[cube]
fn compute_input_coords<EA: Float>(
    output_row: usize,
    output_col: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    #[comptime] options: InterpolateOptions,
) -> (EA, EA) {
    (
        get_input_coord::<EA>(output_row, input_height, output_height, options),
        get_input_coord::<EA>(output_col, input_width, output_width, options),
    )
}

#[cube]
fn get_input_coord<EA: Float>(
    x: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] options: InterpolateOptions,
) -> EA {
    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => match nearest_mode {
            NearestMode::Exact => {
                (EA::cast_from(x) + EA::new(0.5)) * EA::cast_from(input_size)
                    / EA::cast_from(output_size)
            }
            NearestMode::Floor => {
                (EA::cast_from(x) * EA::cast_from(input_size)) / EA::cast_from(output_size)
            }
        },
        _ => {
            if options.align_corners {
                let is_valid_output = (output_size > 1) as usize;
                let safe_denominator = (output_size - 1).max(1);

                EA::cast_from(x * (input_size - 1) * is_valid_output)
                    / EA::cast_from(safe_denominator)
            } else {
                (EA::cast_from(x) + EA::new(0.5)) * EA::cast_from(input_size)
                    / EA::cast_from(output_size)
                    - EA::new(0.5)
            }
        }
    }
}

#[cube]
fn get_reader<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    cube_pos: usize,
    batch: usize,
    vector_index: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> ReaderType<P::EI, N> {
    let vector_size = N::value();

    match blueprint.global {
        GlobalInterpolateBlueprint::GlobalMemoryBlueprint(_global_memory_blueprint) => {
            ReaderType::<P::EI, N>::new_Global(GlobalMemoryReader::new(
                input,
                batch,
                vector_index,
                input_height,
                input_width,
                vector_size,
            ))
        }
        GlobalInterpolateBlueprint::SharedMemoryBlueprint(shared_memory_blueprint) => {
            let halo = comptime!(get_halo(blueprint.options.mode));
            let radius_offset = (halo - 1) / 2;

            let (tile_row, tile_col) = tile_absolute_coords(
                output_width,
                cube_pos,
                0,
                blueprint.tile_size,
                blueprint.options,
            );

            let (tile_mapped_row, tile_mapped_col) = compute_input_coords::<P::EA>(
                tile_row,
                tile_col,
                input_height,
                input_width,
                output_height,
                output_width,
                blueprint.options,
            );

            let (tile_base_row, tile_base_col) = (
                get_value_floor::<P::EA>(tile_mapped_row, blueprint.options),
                get_value_floor::<P::EA>(tile_mapped_col, blueprint.options),
            );

            let min_row = isize::cast_from(tile_base_row) - radius_offset as isize;
            let min_col = isize::cast_from(tile_base_col) - radius_offset as isize;

            ReaderType::new_Shared(SharedMemoryReader::new(
                input,
                batch,
                vector_index,
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
