// use super::layout::{InputTiledLayout, OutputTiledLayout};
use crate::{
    components::mode::{Interpolate, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};
use cubecl::{
    prelude::*,
    std::tensor::{AsView, AsViewMut, View, layout::Coords2d},
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

    fn with_halo(&self, halo: usize) -> TileSize {
        if halo == 0 {
            TileSize {
                h: self.h,
                w: self.w,
            }
        } else {
            // halo is the number of sample taps; the input tile needs to
            // extend by (halo - 1) to cover the extra samples per output
            TileSize {
                h: self.h + (halo - 1),
                w: self.w + (halo - 1),
            }
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
    #[comptime] out_tile_size: TileSize,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let input_val = input[0];
    output[0] = input_val;

    let halo = comptime!(get_halo(options.mode));
    let in_tile_size = out_tile_size.with_halo(halo);
    let smem_num_elements = in_tile_size.total();

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

    // let mut smem = SharedMemory::<Vector<F, N>>::new(smem_num_elements);

    // let out_x_start = CUBE_POS_X as usize * (out_tile_size.w);
    // let out_y_start = CUBE_POS_Y as usize * (out_tile_size.h);

    // let channels = input.shape(3);

    // let batch_idx = CUBE_POS_Z as usize / channels;
    // let channel_idx = CUBE_POS_Z as usize % channels;

    // // 2. Pre-calculate the fixed offsets for this specific Batch and Channel
    // let in_base_offset = (batch_idx * input.stride(0)) + (channel_idx * input.stride(3));
    // let out_base_offset = (batch_idx * output.stride(0)) + (channel_idx * output.stride(3));

    // // let total_units = CUBE_DIM_X as usize * CUBE_DIM_Y as usize;
    // // We assume total units == tile size for simplicity, otherwise we need to loop over the tile in chunks
    // let total_units = out_tile_size.total();
    // let how_many_loads = smem_num_elements.div_ceil(total_units);

    // for i in 0..how_many_loads {
    //     let load_idx = UNIT_POS as usize + (i * total_units);

    //     if load_idx < smem_num_elements {
    //         let local_row = load_idx / in_tile_size.w;
    //         let local_col = load_idx % in_tile_size.w;

    //         let global_in_x = out_x_start + local_col - (halo / 2);
    //         let global_in_y = out_y_start + local_row - (halo / 2);

    //         let scalar_idx =
    //             in_base_offset + (global_in_y * input.stride(1)) + (global_in_x * input.stride(2));

    //         smem[load_idx] = input[scalar_idx / input.vector_size()];
    //     }
    // }

    // sync_cube();

    let local_out_x = UNIT_POS_X as usize;
    let local_out_y = UNIT_POS_Y as usize;

    if local_out_x < out_tile_size.w && local_out_y < out_tile_size.h {
        let frac_x = F::zero();
        let frac_y = F::zero();

        let mut weights_x = Array::<Vector<F, N>>::new(halo);
        let mut weights_y = Array::<Vector<F, N>>::new(halo);

        compute_weights(frac_x, &mut weights_x, options.mode);
        compute_weights(frac_y, &mut weights_y, options.mode);

        // let mut final_value = Vector::<F, N>::cast_from(0.0);

        // for i in 0..halo {
        //     let mut row_interp = Vector::<F, N>::cast_from(0.0);
        //     let smem_row_offset = (local_out_y + i) * in_tile_size.w;

        //     for j in 0..halo {
        //         let pixel = Vector::<F, N>::cast_from(0.0); //smem[smem_row_offset + (local_out_x + j)];
        //         row_interp += pixel * weights_x[j];
        //     }
        //     final_value += row_interp * weights_y[i];
        // }

        // let global_out_x = out_x_start + local_out_x;
        // let global_out_y = out_y_start + local_out_y;

        // let scalar_out_idx =
        //     out_base_offset + (global_out_y * output.stride(1)) + (global_out_x * output.stride(2));

        //output[scalar_out_idx / output.vector_size()] = final_value;
    }
}

#[cube]
fn compute_weights<F: Float, N: Size>(
    frac: F,
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
