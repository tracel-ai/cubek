use crate::{
    components::mode::{Interpolate, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};

use cubecl::prelude::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    h: usize,
    w: usize,
}

impl TileSize {
    pub fn new(h: usize, w: usize) -> Self {
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
    #[comptime] in_tile_size: TileSize,
    #[comptime] out_tile_size: TileSize,
    #[define(F)] _dtype: StorageType,
) {
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    if !local_x < out_tile_size.w && local_y < out_tile_size.h {
        terminate!();
    }

    let out_x = CUBE_POS_X as usize * out_tile_size.w + local_x;
    let out_y = CUBE_POS_Y as usize * out_tile_size.h + local_y;

    if out_x >= output.shape(2) || out_y >= output.shape(1) {
        terminate!();
    }

    let channel_groups = input.shape(3) / input.vector_size();

    let (batch_idx, channel_group_idx) =
        batch_and_channel_group(CUBE_POS_Z as usize, channel_groups);

    let in_base_offset = tensor_base_offset(input, batch_idx, channel_group_idx);

    let out_base_offset = tensor_base_offset(output, batch_idx, channel_group_idx);

    let halo = comptime!(get_halo(options.mode));

    let smem_size = in_tile_size.total_with_halo(halo);

    let mut smem = SharedMemory::<Vector<F, N>>::new(smem_size);

    load_smem(
        input,
        &mut smem,
        in_base_offset,
        out_x,
        out_y,
        in_tile_size,
        options,
    );

    sync_cube();

    let in_x = nearest_in_coord(out_x, input.shape(2), output.shape(2), options);

    let in_y = nearest_in_coord(out_y, input.shape(1), output.shape(1), options);

    let in_idx = tensor_idx(input, in_base_offset, in_y, in_x);

    let out_idx = tensor_idx(output, out_base_offset, out_y, out_x);

    output[out_idx] = input[in_idx];
}

#[cube]
fn load_smem<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    smem: &mut SharedMemory<Vector<F, N>>,
    in_base_offset: usize,
    out_x: usize,
    out_y: usize,
    #[comptime] in_tile_size: TileSize,
    #[comptime] options: InterpolateOptions,
) {
    let halo = comptime!(get_halo(options.mode));

    let in_x_start = out_x.saturating_sub(halo / 2);
    let in_y_start = out_y.saturating_sub(halo / 2);

    let local_idx = UNIT_POS as usize;
    let local_total = CUBE_DIM_X as usize * CUBE_DIM_Y as usize;

    let smem_w = in_tile_size.w + halo;
    let smem_h = in_tile_size.h + halo;
    let smem_size = smem_w * smem_h;

    let load_per_unit = (smem_size + local_total - 1) / local_total;

    for i in 0..load_per_unit {
        let idx = local_idx + i * local_total;
        if idx < smem_size {
            let local_x = idx % smem_w;
            let local_y = idx / smem_w;

            let global_in_x = (in_x_start + local_x).min(input.shape(2) - 1);
            let global_in_y = (in_y_start + local_y).min(input.shape(1) - 1);

            let in_idx = tensor_idx(input, in_base_offset, global_in_y, global_in_x);

            smem[idx] = input[in_idx];
        }
    }
}

#[cube]
fn batch_and_channel_group(z: usize, channel_groups: usize) -> (usize, usize) {
    (z / channel_groups, z % channel_groups)
}

#[cube]
fn tensor_base_offset<F: Float, N: Size>(
    tensor: &Tensor<Vector<F, N>>,
    batch_idx: usize,
    channel_group_idx: usize,
) -> usize {
    (batch_idx * tensor.stride(0) + channel_group_idx * tensor.stride(3)) / tensor.vector_size()
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
fn nearest_in_coord(
    out_coord: usize,
    in_size: usize,
    out_size: usize,
    #[comptime] options: InterpolateOptions,
) -> usize {
    let ratio = get_ratio(in_size, out_size, options);

    let mapped = get_mapped_coord::<f32>(out_coord, ratio, options);

    usize::cast_from(mapped.floor()).min(in_size - 1)
}

#[cube]
fn compute_weights<F: Float, N: Size>(
    frac: f32,
    weights: &mut Array<Vector<F, N>>,
    #[comptime] mode: InterpolateMode,
) {
    match mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::compute_weights(frac, weights),
        _ => todo!(),
    }
}

fn get_halo(mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::halo(),
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
