use cubecl::prelude::*;

use crate::{
    MatrixLayout, StageIdent, SwizzleModes, TileSize,
    tile::{RegisterTile, StridedTile, Tile, scope::Scope},
};

/// Execution mode for the RegisterMatmul
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during load
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during load
    Outer,
}

impl ProductType {
    pub fn from_layouts(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
    ) -> Self {
        let lhs_preferred = match lhs_layout {
            MatrixLayout::RowMajor => ProductType::Inner,
            MatrixLayout::ColMajor => ProductType::Outer,
        };
        let rhs_preferred = match rhs_layout {
            MatrixLayout::RowMajor => ProductType::Outer,
            MatrixLayout::ColMajor => ProductType::Inner,
        };

        if lhs_preferred == rhs_preferred {
            lhs_preferred
        } else if tile_size.m() == 1 {
            rhs_preferred
        } else if tile_size.n() == 1 {
            lhs_preferred
        } else {
            // No better solution
            ProductType::Outer
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub product_type: ProductType,
}

impl RegisterMatmul {
    pub fn new(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
        plane_dim: u32,
        swizzle_modes: SwizzleModes,
    ) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
            product_type: ProductType::from_layouts(lhs_layout, rhs_layout, tile_size),
        }
    }
}

pub(crate) const UNROLL: bool = false;

#[cube]
pub fn register_allocate_lhs<L: Numeric, VL: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<L, VL, Sc, ReadWrite> {
    Tile::new_Register(RegisterTile::<L> {
        data: Array::new((config.tile_size.m() * config.tile_size.k()) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn register_allocate_rhs<R: Numeric, VR: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<R, VR, Sc, ReadWrite> {
    Tile::new_Register(RegisterTile::<R> {
        data: Array::new((config.tile_size.n() * config.tile_size.k()) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn register_allocate_acc<A: Numeric, VA: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
) -> Tile<A, VA, Sc, ReadWrite> {
    Tile::new_Register(RegisterTile::<A> {
        data: Array::new((config.tile_size.m() * config.tile_size.n()) as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn register_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] config: RegisterMatmul,
) {
    let m = config.tile_size.m();
    let n = config.tile_size.n();
    let k = config.tile_size.k();
    match config.product_type {
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

#[cube]
pub fn register_load_from_shared<E: Numeric, ES: Size, N: Numeric, V: Size, IO: SliceVisibility>(
    shared: &StridedTile<E, ES, IO>,
    arr: &mut Array<N>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: RegisterMatmul,
    #[comptime] ident: StageIdent,
) {
    let m = config.tile_size.m();
    let n = config.tile_size.n();
    let k = config.tile_size.k();

    match ident {
        StageIdent::Lhs => match config.product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N, IO>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N, IO>(shared, arr, k, m);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N, IO>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N, IO>(shared, arr, k, m);
                }
            },
        },
        StageIdent::Rhs => match config.product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N, IO>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N, IO>(shared, arr, n, k);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N, IO>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N, IO>(shared, arr, n, k);
                }
            },
        },
        StageIdent::Acc => match matrix_layout {
            MatrixLayout::RowMajor => {
                load_plain::<E, ES, N, IO>(shared, arr, m, n);
            }
            MatrixLayout::ColMajor => {
                load_transposed::<E, ES, N, IO>(shared, arr, n, m);
            }
        },
        _ => panic!("Invalid ident for Register load"),
    }
}

#[cube]
fn load_plain<E: Numeric, ES: Size, N: Numeric, IO: SliceVisibility>(
    tile: &StridedTile<E, ES, IO>,
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
fn load_transposed<E: Numeric, ES: Size, N: Numeric, IO: SliceVisibility>(
    tile: &StridedTile<E, ES, IO>,
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

#[cube]
pub fn register_load_zeros<N: Numeric, V: Size>(
    arr: &mut Array<N>,
    #[comptime] config: RegisterMatmul,
    #[comptime] ident: StageIdent,
) {
    let size = match ident {
        StageIdent::Lhs => config.tile_size.m() * config.tile_size.k(),
        StageIdent::Rhs => config.tile_size.n() * config.tile_size.k(),
        StageIdent::Acc | StageIdent::Out => config.tile_size.m() * config.tile_size.n(),
    };
    for i in 0..size {
        arr[i as usize] = N::from_int(0);
    }
}

#[cube]
pub fn register_write_to_shared<E: Numeric, ES: Size, A: Numeric, VA: Size>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    arr: &Array<A>,
    #[comptime] config: RegisterMatmul,
) {
    let out_vector_size = shared.container.vector_size().comptime() as u32;
    let size_mn = config.tile_size.m() * config.tile_size.n();

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
