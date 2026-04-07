use cubecl::{
    prelude::*,
    {cmma::MmaDefinition, ir::MatrixIdent},
};

use crate::{
    tile::mma::config::{MmaIOConfig, StoreMethod},
    tile::strided_tile::StridedTile,
    {MatrixLayout, as_cmma_layout},
};

/// Writer for storing the output registers.
#[derive(CubeType)]
pub struct MmaStageWriter {}

#[cube]
impl MmaStageWriter {
    pub fn store_fragment<
        E: Numeric,
        N: Size,
        V: Numeric,
        NV: Size,
        A: Numeric,
        B: Numeric,
        CD: Numeric,
    >(
        tile: &mut StridedTile<V, NV, ReadWrite>,
        fragment: &Array<Vector<E, N>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] m: u32,
        #[comptime] config: MmaIOConfig,
    ) {
        let vector_layout = def.vector_layout(ident);
        let transposed = comptime![as_cmma_layout(layout) != vector_layout];

        match config.store_method() {
            StoreMethod::Manual => {
                if transposed {
                    store_manual_transposed(tile, fragment, def, ident, layout);
                } else {
                    store_manual_plain(tile, fragment, def, ident, layout);
                }
            }
            StoreMethod::StoreMatrix => {
                store_stmatrix::<E, N, V, NV, A, B, CD>(tile, fragment, def, transposed, ident, m)
            }
        }
    }
}

#[cube]
fn store_manual_transposed<
    E: Numeric,
    N: Size,
    V: Numeric,
    NV: Size,
    A: Numeric,
    B: Numeric,
    CD: Numeric,
>(
    tile: &mut StridedTile<V, NV, ReadWrite>,
    fragment: &Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let stride = tile.unvectorized_stride();
    let mut tile = tile.with_vector_size::<Const<1>>();

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_vectors {
        #[unroll]
        for n in 0..vector_size {
            let elem_idx = i * vector_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
            let offset = row * stride_row + col * stride_col;
            let offset = tile.stage_offset(offset);

            tile.stage[offset as usize] = Vector::cast_from(fragment[i][n]);
        }
    }
}

#[cube]
fn store_manual_plain<
    E: Numeric,
    N: Size,
    V: Numeric,
    NV: Size,
    A: Numeric,
    B: Numeric,
    CD: Numeric,
>(
    tile: &mut StridedTile<V, NV, ReadWrite>,
    fragment: &Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let lane_id = UNIT_POS_PLANE;
    let stride = tile.unvectorized_stride();
    // Supported on all targets that support manual MMA
    let mut tile = tile.with_vector_size::<N>();

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_vectors {
        let value = fragment[i];
        let elem_idx = i * vector_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
        let offset = row * stride_row + col * stride_col;
        let offset = tile.stage_offset(offset / vector_size as u32);

        tile.stage[offset as usize] = Vector::cast_from(value);
    }
}

/// This is important to use on CUDA because CUDA's matrices are heavily permuted, being organized
/// into 8x8 chunks with only 32 contiguous bits per thread. `stmatrix` uses warp shuffles to move
/// the elements from the mma fragment positions for each thread to 8 consecutive elements in each
/// thread (if executed with x4), then stores them in one transaction. This currently only supports
/// f16, fp8 needs more handling and packed fp4 isn't supported at all. So these currently fall back
/// to manual loading. tf32 isn't supported by the instruction at all.
#[cube]
fn store_stmatrix<
    E: Numeric,
    N: Size,
    V: Numeric,
    NV: Size,
    A: Numeric,
    B: Numeric,
    CD: Numeric,
>(
    tile: &mut StridedTile<V, NV, ReadWrite>,
    fragment: &Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] transposed: bool,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: u32,
) {
    let stage_vector_size = tile.stage.vector_size().comptime();
    let stride = tile.unvectorized_stride();

    let elem_size = E::type_size().comptime();
    let num_regs = def.vectors_per_lane(ident);
    let width = (16 / elem_size / stage_vector_size) as u32;

    let start = stmatrix_offset::<V, A, B, CD>(stride, def, stage_vector_size, ident, m);
    let start = tile.stage_offset(start);

    let mut row_slice = tile
        .stage
        .slice_mut(start as usize, (start + width) as usize);

    let stage_ty = type_of::<V>().comptime();
    let frag_ty = type_of::<E>().comptime();
    if stage_ty == frag_ty {
        def.store_matrix::<Vector<E, NV>, N>(
            &mut row_slice.downcast(),
            fragment,
            ident,
            num_regs,
            transposed,
        );
    } else {
        let mut frag = Array::new(num_regs);
        #[unroll]
        for i in 0..num_regs {
            frag[i] = Vector::cast_from(fragment[i]);
        }
        def.store_matrix::<_, N>(&mut row_slice, &frag, ident, num_regs, transposed);
    }
}

/// Very hardcoded, still haven't figured out the proper generic formula. So keep this separate from
/// the read index for now, and ensure out is row-major.
#[cube]
pub(crate) fn stmatrix_offset<E: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    stride: u32,
    def: MmaDefinition<A, B, CD>,
    #[comptime] stage_vector_size: VectorSize,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: u32,
) -> u32 {
    let (stride_row, stride_col) = (stride, 1);

    let elem_size = E::type_size().comptime();
    let num_regs = def.vectors_per_lane(ident);
    let width = (16 / elem_size) as u32;
    // Height is always 8, and lanes are divided into blocks of 8.
    let height = 8;

    //  Indices are wrapped for < 4 registers.
    let lane = UNIT_POS_PLANE;
    let sub_lane = lane % height;
    let nth_matrix = lane / height % num_regs as u32;

    let tiles_row = m / height;

    // Tiles are arranged in column-major fashion
    let row_offs = (nth_matrix % tiles_row) * 8;
    let col_offs = (nth_matrix / tiles_row) * width;

    let (row, col) = (row_offs + sub_lane, col_offs);

    let start = row * stride_row + col * stride_col;
    start / stage_vector_size as u32
}
