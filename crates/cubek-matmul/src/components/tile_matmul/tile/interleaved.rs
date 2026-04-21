use cubecl::prelude::*;
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::components::tile_matmul::tile::Scope;
use crate::components::tile_matmul::{SharedTileConfig, TileConfig};
use crate::definition::StageIdent;

use super::{InterleavedTile, Tile};

#[cube]
pub fn interleaved_allocate_lhs<L: Numeric, VL: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
) -> Tile<L, VL, Sc, ReadWrite> {
    let m = config.elements_in_tile_m();
    let k = config.elements_in_tile_k();
    let plane_dim = config.plane_dim();
    Tile::new_Interleaved(InterleavedTile::<L> {
        data: Array::new((m * (k / plane_dim)) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn interleaved_allocate_rhs<R: Numeric, VR: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
) -> Tile<R, VR, Sc, ReadWrite> {
    let n = config.elements_in_tile_n();
    let k = config.elements_in_tile_k();
    let plane_dim = config.plane_dim();
    Tile::new_Interleaved(InterleavedTile::<R> {
        data: Array::new(((k / plane_dim) * n) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn interleaved_allocate_acc<A: Numeric, VA: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
) -> Tile<A, VA, Sc, ReadWrite> {
    let m = config.elements_in_tile_m();
    let n = config.elements_in_tile_n();
    Tile::new_Interleaved(InterleavedTile::<A> {
        data: Array::new((m * n) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn interleaved_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    #[comptime] lhs_layout: MatrixLayout,
    rhs: &Array<R>,
    #[comptime] rhs_layout: MatrixLayout,
    acc: &mut Array<A>,
    #[comptime] _acc_layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
) {
    let m = config.elements_in_tile_m() as usize;
    let n = config.elements_in_tile_n() as usize;
    let k = config.elements_in_tile_k() as usize;
    let plane_dim = config.plane_dim() as usize;
    let local_k = k / plane_dim;

    let (lhs_row_count, lhs_col_count) = (m, local_k);
    let (rhs_row_count, rhs_col_count) = (local_k, n);

    #[unroll]
    for m_ in 0..m {
        #[unroll]
        for n_ in 0..n {
            #[unroll]
            for k_ in 0..local_k {
                let lhs_elem = A::cast_from(match lhs_layout {
                    MatrixLayout::RowMajor => lhs[m_ * lhs_col_count + k_],
                    MatrixLayout::ColMajor => lhs[k_ * lhs_row_count + m_],
                });
                let rhs_elem = A::cast_from(match rhs_layout {
                    MatrixLayout::RowMajor => rhs[k_ * rhs_col_count + n_],
                    MatrixLayout::ColMajor => rhs[n_ * rhs_row_count + k_],
                });
                acc[m_ * n + n_] += lhs_elem * rhs_elem;
            }
        }
    }
}

#[cube]
pub fn interleaved_load_from_shared<
    E: Numeric,
    ES: Size,
    N: Numeric,
    V: Size,
    IO: SliceVisibility,
>(
    shared: &StridedTile<E, ES, IO>,
    arr: &mut Array<N>,
    #[comptime] config: SharedTileConfig,
    #[comptime] ident: StageIdent,
) {
    match ident {
        StageIdent::Lhs | StageIdent::Rhs => {
            let m = config.elements_in_tile_m() as usize;
            let n = config.elements_in_tile_n() as usize;
            let k = config.elements_in_tile_k() as usize;
            let plane_dim = config.plane_dim() as usize;
            let k_local = k / plane_dim;

            let shared_layout = comptime!(shared.layout);
            let vector_size = ES::value();

            let unit_id = UNIT_POS_X as usize;
            let k_offset = k_local * unit_id;

            let (strided_dim_count, contiguous_dim_count) = match (shared_layout, ident) {
                (MatrixLayout::RowMajor, StageIdent::Lhs) => (m, k_local),
                (MatrixLayout::RowMajor, StageIdent::Rhs) => (k_local, n),
                (MatrixLayout::ColMajor, StageIdent::Lhs) => (k_local, m),
                (MatrixLayout::ColMajor, StageIdent::Rhs) => (n, k_local),
                _ => unreachable!(),
            };

            let (strided_dim_offset, contiguous_dim_offset) = match (shared_layout, ident) {
                (MatrixLayout::RowMajor, StageIdent::Lhs)
                | (MatrixLayout::ColMajor, StageIdent::Rhs) => (0, k_offset / vector_size),
                (MatrixLayout::RowMajor, StageIdent::Rhs)
                | (MatrixLayout::ColMajor, StageIdent::Lhs) => (k_offset, 0),
                _ => unreachable!(),
            };

            assert!(contiguous_dim_count % vector_size == 0);
            let vector_count_in_dim = contiguous_dim_count / vector_size;

            for i in 0..strided_dim_count {
                for j in 0..vector_count_in_dim {
                    let vector = Vector::<N, ES>::cast_from(shared.get_vector(
                        (i + strided_dim_offset) as u32,
                        (j + contiguous_dim_offset) as u32,
                    ));
                    let vector_start = i * contiguous_dim_count + j * vector_size;
                    for l in 0..vector_size {
                        arr[vector_start + l] = vector[l];
                    }
                }
            }
        }
        StageIdent::Acc => {
            panic!("Not yet implemented: Interleaved acc load from shared");
        }
        _ => panic!("Invalid ident for Interleaved load"),
    }
}

#[cube]
pub fn interleaved_load_zeros<N: Numeric, V: Size>(
    arr: &mut Array<N>,
    #[comptime] config: SharedTileConfig,
) {
    let m = config.elements_in_tile_m() as usize;
    let n = config.elements_in_tile_n() as usize;
    let size = m * n;
    for i in 0..size {
        arr[i] = N::from_int(0);
    }
}

#[cube]
pub fn interleaved_write_to_shared<E: Numeric, ES: Size, A: Numeric, VA: Size>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    arr: &Array<A>,
    #[comptime] config: SharedTileConfig,
) {
    let m = config.elements_in_tile_m();
    let n = config.elements_in_tile_n();
    let out_vector_size = shared.container.vector_size().comptime() as u32;
    let size_mn = m * n;

    // `plane_sum` reduces across the plane, so every unit must participate. Only unit 0 stores.
    #[unroll]
    for i in 0..size_mn / out_vector_size {
        let mut vector = Vector::<A, ES>::empty();
        #[unroll]
        for j in 0..out_vector_size {
            vector[j as usize] = plane_sum(arr[(i * out_vector_size + j) as usize]);
        }
        if UNIT_POS_X == 0 {
            let offs = shared.stage_offset(i);
            shared.container[offs as usize] = Vector::cast_from(vector);
        }
    }
}
