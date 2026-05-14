// use super::layout::{InputTiledLayout, OutputTiledLayout};
use crate::{
    components::mode::{Interpolate, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};
use cubecl::{
    prelude::*,
    // std::tensor::{AsView, AsViewMut, View, layout::Coords2d},
};

/// Number of pixels
/// Total should be divisible by total_units
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    h: usize,
    w: usize,
}

impl TileSize {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h }
    }

    pub fn to_input_tile(&self, ratio_h: f32, ratio_w: f32) -> Self {
        Self::new(
            (self.w as f32 * ratio_w).ceil() as usize,
            (self.h as f32 * ratio_h).ceil() as usize,
        )
    }

    fn with_halo(&self, halo: usize) -> Self {
        Self {
            h: self.h + (halo.max(1) - 1),
            w: self.w + (halo.max(1) - 1),
        }
    }

    fn total(&self) -> usize {
        self.h * self.w
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
    let halo = comptime!(get_halo(options.mode));
    in_tile_size.with_halo(halo);
    let smem_length = in_tile_size.total();

    // let input_layout = InputTiledLayout::new(
    //     out_tile_size.h,
    //     out_tile_size.w,
    //     halo,
    //     input.shape(1) as u32,
    //     input.shape(2) as u32,
    // );
    // let output_layout = OutputTiledLayout::new(
    //     out_tile_size.h,
    //     out_tile_size.w,
    //     output.shape(1) as u32,
    //     output.shape(2) as u32,
    // );

    // Create views
    // let input_view: View<Vector<F, N>, Coords2d> = input.view(input_layout);
    // let mut output_view: View<Vector<F, N>, Coords2d> = output.view_mut(output_layout);

    let mut smem = SharedMemory::<Vector<F, N>>::new(smem_length);

    let out_x_start = CUBE_POS_X as usize * (out_tile_size.w);
    let out_y_start = CUBE_POS_Y as usize * (out_tile_size.h);

    let channels = input.shape(3) / input.vector_size();

    let batch_idx = CUBE_POS_Z as usize / channels;
    let channel_idx = CUBE_POS_Z as usize % channels;

    let in_base_offset = (batch_idx * input.stride(0)) + (channel_idx * input.stride(3));
    let out_base_offset = (batch_idx * output.stride(0)) + (channel_idx * output.stride(3));

    // let total_units = CUBE_DIM_X as usize * CUBE_DIM_Y as usize;
    // We assume total units == tile size for simplicity, otherwise we need to loop over the tile in chunks
    let total_units = out_tile_size.total();
    let load_per_unit = smem_length.div_ceil(total_units);

    for i in 0..load_per_unit {
        let load_idx = UNIT_POS as usize + (i * total_units);

        if load_idx < smem_length {
            let local_row = load_idx / in_tile_size.w as usize;
            let local_col = load_idx % in_tile_size.w as usize;

            let global_in_x = out_x_start + local_col - (halo / 2);
            let global_in_y = out_y_start + local_row - (halo / 2);

            let scalar_idx =
                in_base_offset + (global_in_y * input.stride(1)) + (global_in_x * input.stride(2));

            smem[load_idx] = input[scalar_idx / input.vector_size()];
        }
    }

    sync_cube();

    let local_out_x = UNIT_POS_X as usize;
    let local_out_y = UNIT_POS_Y as usize;

    if local_out_x < out_tile_size.w && local_out_y < out_tile_size.h {
        let ratio_w = get_ratio(input.shape(2), output.shape(2), options);
        let ratio_h = get_ratio(input.shape(1), output.shape(1), options);

        let mapped_x = get_mapped_coord::<F>(local_out_x, ratio_w, options);
        let mapped_y = get_mapped_coord::<F>(local_out_y, ratio_h, options);

        let base_x = mapped_x.floor();
        let base_y = mapped_y.floor();

        let frac_x = mapped_x - base_x;
        let frac_y = mapped_y - base_y;

        let mut weights_x = Array::<Vector<F, N>>::new(halo);
        let mut weights_y = Array::<Vector<F, N>>::new(halo);
        compute_weights(frac_x, &mut weights_x, options.mode);
        compute_weights(frac_y, &mut weights_y, options.mode);

        let mut final_value = Vector::<F, N>::zeroed();

        for i in 0..halo {
            let mut row_interp = Vector::<F, N>::zeroed();
            let smem_row_offset = (base_y as usize + i) * in_tile_size.w;

            for j in 0..halo {
                let pixel = smem[smem_row_offset + (base_x as usize + j)];
                row_interp += pixel * weights_x[j];
            }
            final_value += row_interp * weights_y[i];
        }

        let global_out_x = out_x_start + local_out_x;
        let global_out_y = out_y_start + local_out_y;

        let scalar_out_idx =
            out_base_offset + (global_out_y * output.stride(1)) + (global_out_x * output.stride(2));

        output[scalar_out_idx / output.vector_size()] = final_value;
    }
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
    // Do not "fix": Bug-for-bug compatibility with PyTorch's default nearest-neighbor interpolation.
    if options.mode == InterpolateMode::Nearest {
        f32::cast_from(x as f32 * ratio)
    } else if options.align_corners {
        f32::cast_from(x as f32 * ratio)
    } else {
        f32::cast_from((x as f32 + 0.5) * ratio - 0.5)
    }
}
