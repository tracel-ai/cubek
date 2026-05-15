use crate::{
    components::mode::{Bilinear, Interpolate, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};

use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    h: usize,
    w: usize,
}

impl TileSize {
    pub fn new(w: usize, h: usize) -> Self {
        Self { h, w }
    }

    pub fn from_output_tile(output_tile: Self, ratio_h: f32, ratio_w: f32) -> Self {
        Self::new(
            (output_tile.h as f32 * ratio_h).ceil() as usize,
            (output_tile.w as f32 * ratio_w).ceil() as usize,
        )
    }

    pub fn total_with_halo(self, halo: usize) -> usize {
        (self.h + halo) * (self.w + halo)
    }
}

#[cube(launch, address_type = "dynamic")]
pub fn interpolate_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    #[comptime] options: InterpolateOptions,
    #[comptime] out_tile_size: TileSize,
    #[define(F)] _dtype: StorageType,
) {
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    if !(local_x < out_tile_size.w && local_y < out_tile_size.h) {
        terminate!();
    }

    let channel_groups = input.shape(3) / input.vector_size();
    let (channel_group_idx, global_x) =
        local_x_and_channel_group(CUBE_POS_X as usize, channel_groups);

    let out_x = global_x * out_tile_size.w + local_x;
    let out_y = CUBE_POS_Y as usize * out_tile_size.h + local_y;

    if out_x >= output.shape(2) || out_y >= output.shape(1) {
        terminate!();
    }

    let batch_idx = CUBE_POS_Z as usize;

    let in_base_offset = tensor_base_offset(input, batch_idx, channel_group_idx);

    let out_base_offset = tensor_base_offset(output, batch_idx, channel_group_idx);

    let halo = comptime!(get_halo(options.mode));

    let (in_x, frac_x) = in_coord_mapping(out_x, input.shape(2), output.shape(2), options);
    let (in_y, frac_y) = in_coord_mapping(out_y, input.shape(1), output.shape(1), options);

    let mut weights_x = Array::<Vector<F, N>>::new(halo);
    let mut weights_y = Array::<Vector<F, N>>::new(halo);

    compute_weights(frac_x, &mut weights_x, options.mode);
    compute_weights(frac_y, &mut weights_y, options.mode);

    let mut final_value = Vector::<F, N>::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_interp = Vector::<F, N>::zeroed();
        #[unroll]
        for j in 0..halo {
            let target_y = in_y + (i as isize);
            let target_x = in_x + (j as isize);

            let clamped_y = target_y.max(0).min(input.shape(1) as isize - 1);
            let clamped_x = target_x.max(0).min(input.shape(2) as isize - 1);

            let in_idx = tensor_idx(
                input,
                in_base_offset,
                clamped_y as usize,
                clamped_x as usize,
            );

            row_interp += input[in_idx] * weights_x[j];
        }
        final_value += row_interp * weights_y[i];
    }

    let out_idx = tensor_idx(output, out_base_offset, out_y, out_x);

    output[out_idx] = final_value;
}

#[cube]
fn local_x_and_channel_group(x: usize, channel_groups: usize) -> (usize, usize) {
    (x % channel_groups, x / channel_groups)
}

#[cube]
fn tensor_base_offset<F: Float, N: Size>(
    tensor: &Tensor<Vector<F, N>>,
    batch_idx: usize,
    channel_group_idx: usize,
) -> usize {
    batch_idx * tensor.stride(0) / tensor.vector_size() + channel_group_idx * tensor.stride(3)
}

#[cube]
fn tensor_spatial_offset<F: Float, N: Size>(
    tensor: &Tensor<Vector<F, N>>,
    y: usize,
    x: usize,
) -> usize {
    (y * tensor.stride(1) + x * tensor.stride(2)) / tensor.vector_size()
}

#[cube]
fn tensor_idx<F: Float, N: Size>(
    tensor: &Tensor<Vector<F, N>>,
    base_offset: usize,
    y: usize,
    x: usize,
) -> usize {
    base_offset + tensor_spatial_offset(tensor, y, x)
}

#[cube]
fn in_coord_mapping(
    out_coord: usize,
    in_size: usize,
    out_size: usize,
    #[comptime] options: InterpolateOptions,
) -> (isize, f32) {
    let ratio = get_ratio(in_size, out_size, options);
    let mapped = get_mapped_coord::<f32>(out_coord, ratio, options);

    let mapped_floor = mapped.floor();

    (isize::cast_from(mapped_floor), mapped - mapped_floor)
}

#[cube]
fn compute_weights<F: Float, N: Size>(
    frac: f32,
    weights: &mut Array<Vector<F, N>>,
    #[comptime] mode: InterpolateMode,
) {
    match mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::compute_weights(frac, weights),
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::compute_weights(frac, weights),
        _ => todo!(),
    }
}

fn get_halo(mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::halo(),
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::halo(),
        _ => todo!(),
    }
}

#[cube]
fn get_ratio(
    input_size: usize,
    output_size: usize,
    #[comptime] options: InterpolateOptions,
) -> f32 {
    if options.align_corners && options.mode != InterpolateMode::Nearest {
        f32::cast_from((input_size - 1).max(0)) / f32::cast_from((output_size - 1).max(1))
    } else {
        f32::cast_from(input_size) / f32::cast_from(output_size)
    }
}

#[cube]
fn get_mapped_coord<F: Float>(
    x: usize,
    ratio: f32,
    #[comptime] options: InterpolateOptions,
) -> f32 {
    // PyTorch-compatible nearest behavior
    if options.mode == InterpolateMode::Nearest {
        x as f32 * ratio
    } else if options.align_corners {
        x as f32 * ratio
    } else {
        (x as f32 + 0.5) * ratio - 0.5
    }
}
