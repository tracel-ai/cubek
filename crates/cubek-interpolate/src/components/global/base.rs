use crate::{
    components::{
        readers::{GlobalMemoryReader, ReaderType, SharedMemoryReader},
        writers::Writer,
    },
    definition::{
        InterpolateMode, InterpolateOptions, InterpolatePrecision, NearestMode, compute_weights,
        get_halo, tile_absolute_coords,
    },
    routines::{GlobalInterpolateBlueprint, InterpolateBlueprint},
};
use cubecl::{prelude::*, std::FastDivmod};

#[cube]
pub fn execute_interpolate<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    output: &mut Tensor<Vector<P::EI, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] blueprint: InterpolateBlueprint,
) {
    let (batch, cube_pos, unit_pos, channel_group) = decompose_index(ABSOLUTE_POS, cube_shape);

    let (output_height, output_width) = (output.shape(1), output.shape(2));
    let (input_height, input_width) = (input.shape(1), input.shape(2));

    let (output_row, output_col) =
        tile_absolute_coords(output_width, cube_pos, unit_pos, blueprint.tile_size);

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

    let (weights_row, weights_col) = (
        compute_weights(frac_row, blueprint.options),
        compute_weights(frac_col, blueprint.options),
    );

    let vector_size = N::value();

    let reader = get_reader::<P, N>(
        input,
        cube_pos,
        batch,
        channel_group,
        input_height,
        input_width,
        output_height,
        output_width,
        blueprint,
    );

    let final_value = compute_value_reader::<P, N>(
        input,
        input_height,
        input_width,
        isize::cast_from(input_row_floor),
        isize::cast_from(input_col_floor),
        weights_row,
        weights_col,
        reader,
        blueprint,
    );

    if output_col < output_width && output_row < output_height {
        Writer::write(
            output,
            batch,
            channel_group,
            output_row,
            output_col,
            vector_size,
            final_value,
        );
    }
}

#[cube]
fn decompose_index(
    index: usize,
    cube_shape: Sequence<FastDivmod<usize>>,
) -> (usize, usize, usize, usize) {
    let (rem, channel_group) = cube_shape[0].div_mod(index);
    let (rem, unit_pos) = cube_shape[1].div_mod(rem);
    let (batch, cube_pos) = cube_shape[2].div_mod(rem);
    (batch, cube_pos, unit_pos, channel_group)
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
    channel_group: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> ReaderType<P::EA, N> {
    let vector_size = N::value();

    match blueprint.global {
        GlobalInterpolateBlueprint::GlobalMemoryBlueprint(_global_memory_blueprint) => {
            ReaderType::new_Global(GlobalMemoryReader::new(
                input,
                batch,
                channel_group,
                input_height,
                input_width,
                vector_size,
            ))
        }
        GlobalInterpolateBlueprint::SharedMemoryBlueprint(shared_memory_blueprint) => {
            let halo = comptime!(get_halo(blueprint.options.mode));
            let radius_offset = (halo - 1) / 2;

            let (tile_row, tile_col) =
                tile_absolute_coords(output_width, cube_pos, 0, blueprint.tile_size);

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
                channel_group,
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
fn compute_value_reader<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_height: usize,
    input_width: usize,
    base_row: isize,
    base_col: isize,
    weights_row: Array<Vector<P::EA, N>>,
    weights_col: Array<Vector<P::EA, N>>,

    reader: ReaderType<P::EA, N>,
    #[comptime] blueprint: InterpolateBlueprint,
) -> Vector<P::EI, N> {
    let halo = comptime!(get_halo(blueprint.options.mode));
    let radius_offset = (halo - 1) / 2;

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::zeroed();

        let y = base_row + i as isize - radius_offset as isize;

        #[unroll]
        for j in 0..halo {
            let x = base_col + j as isize - radius_offset as isize;

            let is_in_bounds = is_in_bounds(x, input_width, blueprint.options)
                && is_in_bounds(y, input_height, blueprint.options);

            let clamped_row = y.max(0).min(input_height as isize - 1) as usize;
            let clamped_col = x.max(0).min(input_width as isize - 1) as usize;
            let weight_col = weights_col[j];

            row_value += select(
                is_in_bounds,
                reader.read_weighted::<P::EI>(input, clamped_row, clamped_col, weight_col),
                Vector::zeroed(),
            );
            row_weight_sum += select(is_in_bounds, weight_col, Vector::zeroed());
        }

        let weight_row = weights_row[i];
        final_value += row_value * weight_row;
        total_weight += row_weight_sum * weight_row;
    }

    let epsilon = Vector::cast_from(P::EA::new(1e-7));

    Vector::cast_from(final_value / total_weight.max(epsilon))
}

#[cube]
fn get_value_floor<EA: Float>(value: EA, #[comptime] options: InterpolateOptions) -> EA {
    let float_precision = EA::EPSILON;
    match options.mode {
        InterpolateMode::Nearest(_) => (value + float_precision).floor(),
        _ => value.floor(),
    }
}

// Only used for bounds checking in Lanczos3 mode.
#[cube]
fn is_in_bounds(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Lanczos3 => value >= 0 && value < size as isize,
        _ => true,
    }
}
