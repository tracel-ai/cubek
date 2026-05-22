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

    let (output_width, output_height) = (output.shape(2), output.shape(1));
    let (input_width, input_height) = (input.shape(2), input.shape(1));

    let (x, y) = tile_absolute_coords(output_width, cube_pos, unit_pos, blueprint.tile_size);

    let (mapped_x, mapped_y) = compute_input_coords::<P::EA>(
        x,
        y,
        input_width,
        input_height,
        output_width,
        output_height,
        blueprint.options,
    );

    let (base_x, base_y) = (
        get_mapped_floor::<P::EA>(mapped_x, blueprint.options),
        get_mapped_floor::<P::EA>(mapped_y, blueprint.options),
    );

    let (frac_x, frac_y) = (mapped_x - base_x, mapped_y - base_y);

    let (weights_x, weights_y) = (
        compute_weights(frac_x, blueprint.options),
        compute_weights(frac_y, blueprint.options),
    );

    let vector_size = N::value();

    let reader = get_reader::<P, N>(
        input,
        cube_pos,
        batch,
        channel_group,
        input_width,
        input_height,
        output_width,
        output_height,
        blueprint,
    );

    let final_value = compute_value_reader::<P, N>(
        input,
        input_width,
        input_height,
        isize::cast_from(base_x),
        isize::cast_from(base_y),
        weights_x,
        weights_y,
        reader,
        blueprint,
    );

    if x < output_width && y < output_height {
        let writer = Writer::new();
        writer.write(output, batch, channel_group, x, y, vector_size, final_value);
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
    x: usize,
    y: usize,
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    #[comptime] options: InterpolateOptions,
) -> (EA, EA) {
    let mapped_x = get_input_coord::<EA>(x, input_width, output_width, options);
    let mapped_y = get_input_coord::<EA>(y, input_height, output_height, options);
    (mapped_x, mapped_y)
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
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    #[comptime] blueprint: InterpolateBlueprint,
) -> ReaderType<P::EA, N> {
    let vector_size = N::value();

    match blueprint.global {
        GlobalInterpolateBlueprint::GlobalMemoryBlueprint(_global_memory_blueprint) => {
            ReaderType::new_Global(GlobalMemoryReader::new(
                input,
                batch,
                channel_group,
                input_width,
                input_height,
                vector_size,
            ))
        }
        GlobalInterpolateBlueprint::SharedMemoryBlueprint(shared_memory_blueprint) => {
            let halo = comptime!(get_halo(blueprint.options.mode));
            let radius_offset = (halo - 1) / 2;

            let (tile_x, tile_y) =
                tile_absolute_coords(output_width, cube_pos, 0, blueprint.tile_size);

            let (tile_mapped_x, tile_mapped_y) = compute_input_coords::<P::EA>(
                tile_x,
                tile_y,
                input_width,
                input_height,
                output_width,
                output_height,
                blueprint.options,
            );

            let (tile_base_x, tile_base_y) = (
                get_mapped_floor::<P::EA>(tile_mapped_x, blueprint.options),
                get_mapped_floor::<P::EA>(tile_mapped_y, blueprint.options),
            );

            let min_x = isize::cast_from(tile_base_x) - radius_offset as isize;
            let min_y = isize::cast_from(tile_base_y) - radius_offset as isize;

            ReaderType::new_Shared(SharedMemoryReader::new(
                input,
                batch,
                channel_group,
                input_width,
                input_height,
                min_x,
                min_y,
                vector_size,
                shared_memory_blueprint,
            ))
        }
    }
}

#[cube]
fn compute_value_reader<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_width: usize,
    input_height: usize,
    base_x: isize,
    base_y: isize,
    weights_x: Array<Vector<P::EA, N>>,
    weights_y: Array<Vector<P::EA, N>>,
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

        let y = base_y + i as isize - radius_offset as isize;

        #[unroll]
        for j in 0..halo {
            let x = base_x + j as isize - radius_offset as isize;

            let is_in_bounds = is_in_bounds(x, input_width, blueprint.options)
                && is_in_bounds(y, input_height, blueprint.options);

            let clamped_y = y.max(0).min(input_height as isize - 1) as usize;
            let clamped_x = x.max(0).min(input_width as isize - 1) as usize;
            let weight_x = weights_x[j];

            row_value += select(
                is_in_bounds,
                reader.read_weighted::<P::EI>(input, clamped_x, clamped_y, weight_x),
                Vector::zeroed(),
            );
            row_weight_sum += select(is_in_bounds, weight_x, Vector::zeroed());
        }

        let weight_y = weights_y[i];
        final_value += row_value * weight_y;
        total_weight += row_weight_sum * weight_y;
    }

    let epsilon = Vector::cast_from(P::EA::new(1e-7));

    Vector::cast_from(final_value / total_weight.max(epsilon))
}

#[cube]
fn get_mapped_floor<EA: Float>(mapped: EA, #[comptime] options: InterpolateOptions) -> EA {
    let float_precision = EA::EPSILON;
    match options.mode {
        InterpolateMode::Nearest(_) => (mapped + float_precision).floor(),
        _ => mapped.floor(),
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
