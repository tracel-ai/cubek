use crate::{
    components::{
        global::{TileSize, tile_absolute_coords},
        readers::Reader,
        writers::Writer,
    },
    definition::{InterpolateMode, InterpolateOptions, NearestMode, compute_weights, get_halo},
};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn interpolate_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] output_tile_size: TileSize,
    #[comptime] options: InterpolateOptions,
    #[define(F)] _dtype: StorageType,
) {
    let (batch, cube_pos, unit_pos, channel_group) = decompose_index(ABSOLUTE_POS, cube_shape);

    let (output_width, output_height) = (output.shape(2), output.shape(1));
    let (input_width, input_height) = (input.shape(2), input.shape(1));

    let (x, y) = tile_absolute_coords(output_width, cube_pos, unit_pos, output_tile_size);

    if x >= output_width || y >= output_height {
        terminate!();
    }

    let (mapped_x, mapped_y) = compute_input_coords::<F>(
        x,
        y,
        input_width,
        input_height,
        output_width,
        output_height,
        options,
    );

    let base_x_floor = mapped_x.floor();
    let base_y_floor = mapped_y.floor();

    let (frac_x, frac_y) = (mapped_x - base_x_floor, mapped_y - base_y_floor);

    let (base_x, base_y) = (
        isize::cast_from(base_x_floor),
        isize::cast_from(base_y_floor),
    );

    let (weights_x, weights_y) = (
        compute_weights(frac_x, options),
        compute_weights(frac_y, options),
    );

    let vector_size = input.vector_size();

    let final_value = compute_value(
        input,
        batch,
        channel_group,
        vector_size,
        input_width,
        input_height,
        base_x,
        base_y,
        weights_x,
        weights_y,
        options,
    );

    let writer = Writer::new(channel_group);

    writer.write(output, batch, x, y, vector_size, final_value);
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
fn compute_input_coords<F: Float>(
    x: usize,
    y: usize,
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    #[comptime] options: InterpolateOptions,
) -> (F, F) {
    let mapped_x = get_input_coord::<F>(x, input_width, output_width, options);
    let mapped_y = get_input_coord::<F>(y, input_height, output_height, options);
    (mapped_x, mapped_y)
}

#[cube]
fn get_input_coord<F: Float>(
    x: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] options: InterpolateOptions,
) -> F {
    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => match nearest_mode {
            NearestMode::Exact => {
                (F::cast_from(x) + F::new(0.5)) * F::cast_from(input_size)
                    / F::cast_from(output_size)
            }
            NearestMode::Floor => {
                (F::cast_from(x) * F::cast_from(input_size)) / F::cast_from(output_size)
            }
        },
        _ => {
            if options.align_corners {
                let is_valid_output = (output_size > 1) as usize;
                let safe_denominator = (output_size - 1).max(1);

                F::cast_from(x * (input_size - 1) * is_valid_output)
                    / F::cast_from(safe_denominator)
            } else {
                (F::cast_from(x) + F::new(0.5)) * F::cast_from(input_size)
                    / F::cast_from(output_size)
                    - F::new(0.5)
            }
        }
    }
}

#[cube]
fn compute_value<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    batch: usize,
    channel_group: usize,
    vector_size: usize,
    input_width: usize,
    input_height: usize,
    base_x: isize,
    base_y: isize,
    weights_x: Array<Vector<F, N>>,
    weights_y: Array<Vector<F, N>>,
    #[comptime] options: InterpolateOptions,
) -> Vector<F, N> {
    let input_offset = batch * input.stride(0);
    let reader = Reader::new(channel_group);

    let halo = comptime!(get_halo(options.mode));
    let radius_offset = (halo - 1) / 2;

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::zeroed();

        let unclamped_y = base_y + i as isize - radius_offset as isize;
        let y = unclamped_y.max(0).min(input_height as isize - 1) as usize;
        let row_offset = input_offset + y * input.stride(1);

        #[unroll]
        for j in 0..halo {
            let unclamped_x = base_x + j as isize - radius_offset as isize;
            let x = unclamped_x.max(0).min(input_width as isize - 1) as usize;

            let is_in_bounds = is_in_bounds(unclamped_x, input_width, options)
                && is_in_bounds(unclamped_y, input_height, options);
            let weight_x = weights_x[j];

            row_value += select(
                is_in_bounds,
                reader.read_weighted(input, row_offset, x, vector_size, weight_x),
                Vector::zeroed(),
            );
            row_weight_sum += select(is_in_bounds, weight_x, Vector::zeroed());
        }

        let weight_y = weights_y[i];
        final_value += row_value * weight_y;
        total_weight += row_weight_sum * weight_y;
    }

    let epsilon = Vector::cast_from(F::new(1e-7));

    final_value / total_weight.max(epsilon)
}

// Only used for bounds checking in Lanczos3 mode.
#[cube]
fn is_in_bounds(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Lanczos3 => value >= 0 && value < size as isize,
        _ => true,
    }
}
