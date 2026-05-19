use crate::{
    components::mode::{Bicubic, Bilinear, Interpolate, Lanczos3, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn interpolate_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    tile_width: usize,
    tile_height: usize,
    #[comptime] options: InterpolateOptions,
    #[define(F)] _dtype: StorageType,
) {
    let (rem, channel_group) = cube_shape[0].div_mod(ABSOLUTE_POS);
    let (rem, unit_pos) = cube_shape[1].div_mod(rem);
    let (batch, cube_pos) = cube_shape[2].div_mod(rem);

    let (output_width, output_height) = (output.shape(2), output.shape(1));

    let (x, y) = compute_absolute_coords(
        tile_width,
        tile_height,
        output_width,
        cube_pos,
        unit_pos,
        options,
    );

    if x >= output_width || y >= output_height {
        terminate!();
    }

    let (input_width, input_height) = (input.shape(2), input.shape(1));

    let (mapped_x, mapped_y) = compute_input_coords::<F, N>(
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

    let (weights_x, weights_y) = compute_weights(frac_x, frac_y, options);

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

    let out_index = (batch * output.stride(0) + y * output.stride(1) + x * output.stride(2))
        / vector_size
        + channel_group * output.stride(3);

    output[out_index] = final_value;
}

#[cube]
fn compute_absolute_coords(
    tile_width: usize,
    tile_height: usize,
    output_width: usize,
    cube_pos: usize,
    unit_pos: usize,
    #[comptime] options: InterpolateOptions,
) -> (usize, usize) {
    if comptime!(is_row_vector(options)) {
        let flat = cube_pos * tile_width + unit_pos;
        (flat % output_width, flat / output_width)
    } else {
        let num_tiles_x = output_width.div_ceil(tile_width);

        let (local_x, local_y) = local_coords(tile_width, unit_pos, options);
        let (cube_x, cube_y) = cube_coords(cube_pos, num_tiles_x);

        (
            cube_x * tile_width + local_x,
            cube_y * tile_height + local_y,
        )
    }
}

#[cube]
fn local_coords(
    tile_width: usize,
    unit_pos: usize,
    #[comptime] options: InterpolateOptions,
) -> (usize, usize) {
    if comptime!(is_row_vector(options)) {
        (unit_pos, 0)
    } else {
        (unit_pos % tile_width, unit_pos / tile_width)
    }
}

#[cube]
fn cube_coords(cube_pos: usize, num_tiles_x: usize) -> (usize, usize) {
    (cube_pos % num_tiles_x, cube_pos / num_tiles_x)
}

#[cube]
fn compute_input_coords<F: Float, N: Size>(
    x: usize,
    y: usize,
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    #[comptime] options: InterpolateOptions,
) -> (F, F) {
    let mapped_x = get_input_coord::<F, N>(x, input_width, output_width, options);
    let mapped_y = get_input_coord::<F, N>(y, input_height, output_height, options);
    (mapped_x, mapped_y)
}

#[cube]
fn get_input_coord<F: Float, N: Size>(
    x: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] options: InterpolateOptions,
) -> F {
    if options.mode == InterpolateMode::Nearest {
        // Do not "fix": Bug-for-bug compatibility with PyTorch's default nearest-neighbor interpolation.
        (F::cast_from(x) * F::cast_from(input_size)) / F::cast_from(output_size)
    } else if options.align_corners {
        F::cast_from(x) * F::cast_from(input_size - 1).max(F::zero())
            / F::cast_from(output_size - 1).max(F::one())
    } else {
        (F::cast_from(x) + F::new(0.5)) * F::cast_from(input_size) / F::cast_from(output_size)
            - F::new(0.5)
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

    let halo = comptime!(get_halo(options));
    let radius_offset = (halo - 1) / 2;

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::zeroed();

        let unclamped_y = base_y + i as isize - radius_offset as isize;
        let y = clamp(unclamped_y, input_height, options);
        let row_offset = input_offset + y as usize * input.stride(1);

        if is_in_bounds(unclamped_y, input_height, options) {
            #[unroll]
            for j in 0..halo {
                let unclamped_x = base_x + j as isize - radius_offset as isize;
                let x = clamp(unclamped_x, input_width, options);

                let is_in_bounds = is_in_bounds(unclamped_x, input_width, options)
                    && is_in_bounds(unclamped_y, input_height, options);
                let weight_x = weights_x[j];

                row_value += select(
                    is_in_bounds,
                    get_input_value(
                        input,
                        row_offset,
                        x as usize,
                        vector_size,
                        channel_group,
                        weight_x,
                    ),
                    Vector::zeroed(),
                );
                row_weight_sum += select(is_in_bounds, weight_x, Vector::zeroed());
            }

            let weight_y = weights_y[i];
            final_value += row_value * weight_y;
            total_weight += row_weight_sum * weight_y;
        }
    }

    let epsilon = Vector::cast_from(F::new(1e-6f32));

    final_value / total_weight.max(epsilon)
}

#[cube]
fn get_input_value<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    row_offset: usize,
    column_offset: usize,
    vector_size: usize,
    channel_group: usize,
    weight: Vector<F, N>,
) -> Vector<F, N> {
    let input_index = (row_offset + column_offset * input.stride(2)) / vector_size
        + channel_group * input.stride(3);

    let pixel = input[input_index];
    pixel * weight
}

#[cube]
fn compute_weights<F: Float, N: Size>(
    frac_x: F,
    frac_y: F,
    #[comptime] options: InterpolateOptions,
) -> (Array<Vector<F, N>>, Array<Vector<F, N>>) {
    match options.mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::compute_weights(frac_x, frac_y),
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::compute_weights(frac_x, frac_y),
        InterpolateMode::Bicubic => <Bicubic as Interpolate>::compute_weights(frac_x, frac_y),
        InterpolateMode::Lanczos3 => <Lanczos3 as Interpolate>::compute_weights(frac_x, frac_y),
    }
}

#[cube]
fn clamp(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> isize {
    match options.mode {
        InterpolateMode::Bilinear | InterpolateMode::Bicubic => value.max(0).min(size as isize - 1),
        _ => value,
    }
}

#[cube]
fn is_in_bounds(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Lanczos3 => value >= 0 && value < size as isize,
        _ => true,
    }
}

fn is_row_vector(options: InterpolateOptions) -> bool {
    options.mode == InterpolateMode::Nearest
}

fn get_halo(options: InterpolateOptions) -> usize {
    match options.mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::halo(),
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::halo(),
        InterpolateMode::Bicubic => <Bicubic as Interpolate>::halo(),
        InterpolateMode::Lanczos3 => <Lanczos3 as Interpolate>::halo(),
    }
}
