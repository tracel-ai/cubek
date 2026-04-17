use cubecl::prelude::*;
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::components::tile::{ProductType, SharedTileConfig, TileConfig};
use crate::definition::StageIdent;

use super::{RegisterTile, Tilex};

pub(crate) const UNROLL: bool = false;

// ===========================================================================
// Allocate
// ===========================================================================

#[cube]
pub fn register_allocate_lhs<L: Numeric, VL: Size>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] product_type: ProductType,
) -> Tilex<L, VL, ReadWrite> {
    Tilex::new_Register(RegisterTile::<L> {
        data: Array::new((config.elements_in_tile_m() * config.elements_in_tile_k()) as usize),
        matrix_layout: layout,
        config,
        product_type,
    })
}

#[cube]
pub fn register_allocate_rhs<R: Numeric, VR: Size>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] product_type: ProductType,
) -> Tilex<R, VR, ReadWrite> {
    Tilex::new_Register(RegisterTile::<R> {
        data: Array::new((config.elements_in_tile_n() * config.elements_in_tile_k()) as usize),
        matrix_layout: layout,
        config,
        product_type,
    })
}

#[cube]
pub fn register_allocate_acc<A: Numeric, VA: Size>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] product_type: ProductType,
) -> Tilex<A, VA, ReadWrite> {
    Tilex::new_Register(RegisterTile::<A> {
        data: Array::new((config.elements_in_tile_m() * config.elements_in_tile_n()) as usize),
        matrix_layout: layout,
        config,
        product_type,
    })
}

// ===========================================================================
// Execute: (Register, Register, Register)
// ===========================================================================

#[cube]
pub fn register_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] config: SharedTileConfig,
    #[comptime] product_type: ProductType,
) {
    let m = config.elements_in_tile_m();
    let n = config.elements_in_tile_n();
    let k = config.elements_in_tile_k();
    match product_type {
        ProductType::Inner => {
            inner_product::<L, R, A>(lhs, rhs, acc, m, n, k);
        }
        ProductType::Outer => {
            outer_product::<L, R, A>(lhs, rhs, acc, m, n, k);
        }
    }
}

#[cube]
fn inner_product<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    #[unroll(UNROLL)]
    for m_ in 0..m as usize {
        #[unroll(UNROLL)]
        for n_ in 0..n as usize {
            #[unroll(UNROLL)]
            for k_ in 0..k as usize {
                let lhs_elem = A::cast_from(lhs[m_ * k as usize + k_]);
                let rhs_elem = A::cast_from(rhs[n_ * k as usize + k_]);
                acc[m_ * n as usize + n_] += lhs_elem * rhs_elem;
            }
        }
    }
}

#[cube]
fn outer_product<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    #[unroll(UNROLL)]
    for k_ in 0..k as usize {
        #[unroll(UNROLL)]
        for m_ in 0..m as usize {
            let lhs_elem = A::cast_from(lhs[k_ * m as usize + m_]);
            #[unroll(UNROLL)]
            for n_ in 0..n as usize {
                let rhs_elem = A::cast_from(rhs[k_ * n as usize + n_]);
                acc[m_ * n as usize + n_] += lhs_elem * rhs_elem;
            }
        }
    }
}

// ===========================================================================
// Load: SharedMemory -> Register
// ===========================================================================

#[cube]
pub fn register_load_from_shared<E: Numeric, ES: Size, N: Numeric, V: Size>(
    shared: &StridedTile<E, ES, ReadOnly>,
    arr: &mut Array<N>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] product_type: ProductType,
    #[comptime] ident: StageIdent,
) {
    let m = config.elements_in_tile_m();
    let n = config.elements_in_tile_n();
    let k = config.elements_in_tile_k();

    match ident {
        StageIdent::Lhs => match product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N>(shared, arr, k, m);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N>(shared, arr, k, m);
                }
            },
        },
        StageIdent::Rhs => match product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N>(shared, arr, n, k);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N>(shared, arr, n, k);
                }
            },
        },
        StageIdent::Acc => match matrix_layout {
            MatrixLayout::RowMajor => {
                load_plain::<E, ES, N>(shared, arr, m, n);
            }
            MatrixLayout::ColMajor => {
                load_transposed::<E, ES, N>(shared, arr, n, m);
            }
        },
        _ => panic!("Invalid ident for Register load"),
    }
}

#[cube]
fn load_plain<E: Numeric, ES: Size, N: Numeric>(
    tile: &StridedTile<E, ES, ReadOnly>,
    arr: &mut Array<N>,
    #[comptime] num_segments: u32,
    #[comptime] segment_size: u32,
) {
    let line_size = ES::value() as u32;
    let num_lines_per_segment = segment_size / line_size;

    #[unroll(UNROLL)]
    for segment in 0..num_segments {
        #[unroll(UNROLL)]
        for line_within_segment in 0..num_lines_per_segment {
            let line = tile.get_vector(segment, line_within_segment);
            #[unroll]
            for pos_within_line in 0..line_size {
                arr[(segment * segment_size + line_within_segment * line_size + pos_within_line)
                    as usize] = N::cast_from(line[pos_within_line as usize]);
            }
        }
    }
}

#[cube]
fn load_transposed<E: Numeric, ES: Size, N: Numeric>(
    tile: &StridedTile<E, ES, ReadOnly>,
    arr: &mut Array<N>,
    #[comptime] num_segments: u32,
    #[comptime] segment_size: u32,
) {
    let line_size = ES::value() as u32;
    let num_lines_per_segment = segment_size / line_size;

    #[unroll(UNROLL)]
    for segment in 0..num_segments {
        #[unroll(UNROLL)]
        for line_within_segment in 0..num_lines_per_segment {
            let line = tile.get_vector(segment, line_within_segment);
            #[unroll]
            for pos_within_line in 0..line_size {
                arr[((line_within_segment * line_size + pos_within_line) * num_segments + segment)
                    as usize] = N::cast_from(line[pos_within_line as usize]);
            }
        }
    }
}

// ===========================================================================
// Load: None -> Register (zero fill)
// ===========================================================================

#[cube]
pub fn register_load_zeros<N: Numeric, V: Size>(
    arr: &mut Array<N>,
    #[comptime] config: SharedTileConfig,
    #[comptime] ident: StageIdent,
) {
    let size = match ident {
        StageIdent::Lhs => config.elements_in_tile_m() * config.elements_in_tile_k(),
        StageIdent::Rhs => config.elements_in_tile_n() * config.elements_in_tile_k(),
        StageIdent::Acc | StageIdent::Out => {
            config.elements_in_tile_m() * config.elements_in_tile_n()
        }
    };
    for i in 0..size {
        arr[i as usize] = N::from_int(0);
    }
}

// ===========================================================================
// Write: Register -> SharedMemory
// ===========================================================================

#[cube]
pub fn register_write_to_shared<E: Numeric, ES: Size, A: Numeric, VA: Size>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    arr: &Array<A>,
    #[comptime] config: SharedTileConfig,
) {
    let out_vector_size = shared.container.vector_size().comptime() as u32;
    let size_mn = config.elements_in_tile_m() * config.elements_in_tile_n();

    #[unroll(false)]
    for i in 0..size_mn / out_vector_size {
        let offs = shared.stage_offset(i);
        let mut vector = Vector::<A, ES>::empty();
        #[unroll]
        for j in 0..out_vector_size {
            vector[j as usize] = arr[(i * out_vector_size + j) as usize];
        }
        shared.container[offs as usize] = Vector::cast_from(vector);
    }
}
