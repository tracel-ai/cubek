use crate::{
    components::mode::{Interpolate, Nearest},
    definition::{InterpolateMode, InterpolateOptions},
};
use cubecl::prelude::*;

/// Number of pixels
/// Total should be divisible by total_units
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub struct TileSize {
    h: usize,
    w: usize,
}

impl TileSize {
    fn to_haloed(&self, halo: usize) -> TileSize {
        TileSize {
            h: self.h + halo,
            w: self.w + halo,
        }
    }

    fn total(&self) -> usize {
        self.h * self.w
    }
}

#[cube(launch, debug)]
pub fn interpolate_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    #[comptime] options: InterpolateOptions,
    #[comptime] out_tile_size: TileSize,
) {
    let halo = get_halo(options.mode) as usize;
    let in_tile_size_w = (out_tile_size.w as usize) + halo;
    let in_tile_size_h = (out_tile_size.h as usize) + halo;
    let smem_num_elements = in_tile_size_w * in_tile_size_h;

    let mut smem = SharedMemory::<Vector<F, N>>::new(smem_num_elements);

    let out_x_start = CUBE_POS_X as usize * (out_tile_size.w as usize);
    let out_y_start = CUBE_POS_Y as usize * (out_tile_size.h as usize);

    let total_units = CUBE_DIM_X as usize * CUBE_DIM_Y as usize;
    let how_many_loads = smem_num_elements.div_ceil(total_units);

    for i in 0..how_many_loads {
        let load_idx = UNIT_POS as usize + (i * total_units);

        if load_idx < smem_num_elements {
            let local_row = load_idx / in_tile_size_w;
            let local_col = load_idx % in_tile_size_w;

            // Coordonnée globale dans l'image d'entrée
            // On décale de -halo/2 pour centrer le voisinage si nécessaire
            let global_in_x = out_x_start + local_col - (halo / 2);
            let global_in_y = out_y_start + local_row - (halo / 2);

            smem[load_idx] = input[global_in_x * input.stride(2)
                + global_in_y * input.stride(1) / input.vector_size()];
        }
    }

    sync_cube();

    let local_out_x = UNIT_POS_X as usize;
    let local_out_y = UNIT_POS_Y as usize;

    if local_out_x < out_tile_size.w && local_out_y < out_tile_size.h {
        // Dans cet exemple, on assume un ratio 1:1 pour la logique des poids
        // Sinon, calculez sub_x/sub_y basé sur le ratio de redimensionnement
        let sub_x = F::new(0.5);
        let sub_y = F::new(0.5);

        // On utilise des tableaux fixes pour les poids (comptime size)
        let mut weights_x = Array::<Vector<F, N>>::new(halo);
        let mut weights_y = Array::<Vector<F, N>>::new(halo);

        compute_weights(sub_x, &mut weights_x, options.mode);
        compute_weights(sub_y, &mut weights_y, options.mode);

        let mut final_value = Vector::<F, N>::new(F::new(0.0));

        for i in 0..halo {
            let mut row_interp = Vector::<F, N>::new(F::new(0.0));
            let smem_row_offset = (local_out_y + i) * in_tile_size_w;

            for j in 0..halo {
                let pixel = smem[smem_row_offset + (local_out_x + j)];
                row_interp += pixel * weights_x[j];
            }
            final_value += row_interp * weights_y[i];
        }

        let global_out_x = out_x_start + local_out_x;
        let global_out_y = out_y_start + local_out_y;
        output[global_out_y * output.stride(1)
            + global_out_x * output.stride(2) / output.vector_size()] = final_value;
    }
}

#[cube]
fn compute_weights<F: Float, N: Size>(
    _sub: F,
    weights: &mut Array<Vector<F, N>>,
    #[comptime] mode: InterpolateMode,
) {
    match mode {
        InterpolateMode::Nearest => {
            weights[0] = Vector::new(F::new(1.0));
        }
        _ => { /* Bilinear, Lanczos3... */ }
    }
}

#[cube]
fn get_halo(#[comptime] mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest => <Nearest as Interpolate>::halo(),
        _ => todo!(),
    }
}
