use cubecl::prelude::*;
use cubecl::{cmma::MmaDefinition, ir::MatrixIdent};

use crate::components::tile::{
    StridedTile,
    mma::config::{MmaMatmulConfig, StoreMethod},
};
use crate::definition::{MatrixLayout, as_cmma_layout};

/// Writer for storing the output registers.
#[derive(CubeType)]
pub struct MmaStageWriter {}

#[cube]
impl MmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] config: MmaMatmulConfig,
    ) {
        let line_layout = def.line_layout(ident);
        let transposed = comptime![as_cmma_layout(layout) != line_layout];

        match config.store_method() {
            StoreMethod::Manual => {
                if transposed {
                    store_manual_transposed(tile, fragment, def, ident, layout);
                } else {
                    store_manual_plain(tile, fragment, def, ident, layout);
                }
            }
            StoreMethod::StoreMatrix => {
                store_stmatrix::<E, V, A, B, CD>(tile, fragment, def, transposed, ident, config)
            }
        }
    }
}

#[cube]
fn store_manual_transposed<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &mut StridedTile<V, ReadWrite>,
    fragment: &Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let (_, stride) = tile.as_unlined_mut();
    let mut tile = tile.with_line_size(1usize);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        #[unroll]
        for n in 0..line_size {
            let elem_idx = i * line_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
            let offset = row * stride_row + col * stride_col;
            let offset = tile.stage_offset(offset);

            tile.stage[offset as usize] = Line::cast_from(fragment[i][n]);
        }
    }
}

#[cube]
fn store_manual_plain<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &mut StridedTile<V, ReadWrite>,
    fragment: &Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;
    let (_, stride) = tile.as_unlined_mut();
    // Supported on all targets that support manual MMA
    let mut tile = tile.with_line_size(line_size);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let value = fragment[i];
        let elem_idx = i * line_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
        let offset = row * stride_row + col * stride_col;
        let offset = tile.stage_offset(offset / line_size as u32);

        tile.stage[offset as usize] = Line::cast_from(value);
    }
}

/// This is important to use on CUDA because CUDA's matrices are heavily permuted, being organized
/// into 8x8 chunks with only 32 contiguous bits per thread. `stmatrix` uses warp shuffles to move
/// the elements from the mma fragment positions for each thread to 8 consecutive elements in each
/// thread (if executed with x4), then stores them in one transaction. This currently only supports
/// f16, fp8 needs more handling and packed fp4 isn't supported at all. So these currently fall back
/// to manual loading. tf32 isn't supported by the instruction at all.
#[cube]
fn store_stmatrix<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &mut StridedTile<V, ReadWrite>,
    fragment: &Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] transposed: bool,
    #[comptime] ident: MatrixIdent,
    #[comptime] config: MmaMatmulConfig,
) {
    let stage_line_size = tile.stage.line_size().comptime();
    let (_, stride) = tile.as_unlined_mut();

    let elem_size = E::type_size().comptime();
    let num_regs = def.lines_per_lane(ident);
    let width = (16 / elem_size / stage_line_size) as u32;

    let start = stmatrix_offset::<V, A, B, CD>(stride, def, stage_line_size, ident, config);
    let start = tile.stage_offset(start);

    let mut row_slice = tile
        .stage
        .slice_mut(start as usize, (start + width) as usize);

    let stage_ty = type_of::<V>().comptime();
    let frag_ty = type_of::<E>().comptime();
    if stage_ty == frag_ty {
        def.store_matrix(
            &mut row_slice.downcast(),
            fragment,
            ident,
            num_regs,
            transposed,
        );
    } else {
        let mut frag = Array::lined(num_regs, fragment.line_size());
        #[unroll]
        for i in 0..num_regs {
            frag[i] = Line::cast_from(fragment[i]);
        }
        def.store_matrix(&mut row_slice, &frag, ident, num_regs, transposed);
    }
}

/// Very hardcoded, still haven't figured out the proper generic formula. So keep this separate from
/// the read index for now, and ensure out is row-major.
#[cube]
pub(crate) fn stmatrix_offset<E: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    stride: u32,
    def: MmaDefinition<A, B, CD>,
    #[comptime] stage_line_size: LineSize,
    #[comptime] ident: MatrixIdent,
    #[comptime] config: MmaMatmulConfig,
) -> u32 {
    let tiling = config.shared.tile_size;
    let (stride_row, stride_col) = (stride, 1);

    let elem_size = E::type_size().comptime();
    let num_regs = def.lines_per_lane(ident);
    let width = (16 / elem_size) as u32;
    // Height is always 8, and lanes are divided into blocks of 8.
    let height = 8;

    //  Indices are wrapped for < 4 registers.
    let lane = UNIT_POS_PLANE;
    let sub_lane = lane % height;
    let nth_matrix = lane / height % num_regs as u32;

    let tiles_row = tiling.m() / height;

    // Tiles are arranged in column-major fashion
    let row_offs = (nth_matrix % tiles_row) * 8;
    let col_offs = (nth_matrix / tiles_row) * width;

    let (row, col) = (row_offs + sub_lane, col_offs);

    let start = row * stride_row + col * stride_col;
    start / stage_line_size as u32
}
