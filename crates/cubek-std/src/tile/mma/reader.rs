use std::marker::PhantomData;

use cubecl::{
    prelude::*,
    {cmma::MmaDefinition, ir::MatrixIdent},
};

use crate::{
    TileSize,
    tile::mma::config::{LoadMethod, MmaIOConfig},
    tile::strided_tile::StridedTile,
    tile::tile_kind::{Filled, Strided, TileKind},
    {MatrixLayout, as_cmma_layout, from_cmma_layout},
};

/// Generic CMMA reader over any tile type
#[cube]
pub trait MmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<
        E: Numeric,
        N: Size,
        V: Numeric,
        NV: Size,
        A: Numeric,
        B: Numeric,
        CD: Numeric,
    >(
        tile: &<Self::TileKind as TileKind>::Tile<V, NV>,
        fragment: &mut Array<Vector<E, N>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] tile_size: TileSize,
        #[comptime] config: MmaIOConfig,
    );
}

/// Reader to load the manual MMA registers. Tile kind determines implementation.
#[derive(CubeType)]
pub struct MmaStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl MmaFragmentReader for MmaStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<
        E: Numeric,
        NE: Size,
        V: Numeric,
        NV: Size,
        A: Numeric,
        B: Numeric,
        CD: Numeric,
    >(
        tile: &StridedTile<V, NV>,
        fragment: &mut Array<Vector<E, NE>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] tile_size: TileSize,
        #[comptime] config: MmaIOConfig,
    ) {
        let vector_layout = def.vector_layout(ident);

        let transposed = comptime![as_cmma_layout(layout) != vector_layout];

        match config.load_method(ident) {
            LoadMethod::Manual => {
                if transposed {
                    load_manual_transposed(tile, fragment, def, ident, layout);
                } else {
                    load_manual_plain(tile, fragment, def, ident, layout);
                }
            }
            LoadMethod::LoadMatrix => {
                load_ldmatrix(tile, fragment, def, transposed, ident, layout, tile_size);
            }
        }
    }
}

#[cube]
fn load_manual_transposed<
    E: Numeric,
    N: Size,
    V: Numeric,
    NV: Size,
    A: Numeric,
    B: Numeric,
    CD: Numeric,
>(
    tile: &StridedTile<V, NV>,
    fragment: &mut Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let stride = tile.unvectorized_stride();
    let tile = tile.with_vector_size::<Const<1>>();

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_vectors {
        let mut vector = Vector::empty();
        #[unroll]
        for n in 0..vector_size {
            let elem_idx = i * vector_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
            let offset = row * stride_row + col * stride_col;
            let offset = tile.stage_offset(offset);

            vector[n] = E::cast_from(tile.container[offset as usize]);
        }
        fragment[i] = vector;
    }
}

#[cube]
fn load_manual_plain<
    E: Numeric,
    N: Size,
    V: Numeric,
    NV: Size,
    A: Numeric,
    B: Numeric,
    CD: Numeric,
>(
    tile: &StridedTile<V, NV>,
    fragment: &mut Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = N::value();

    let lane_id = UNIT_POS_PLANE;
    let stride = tile.unvectorized_stride();
    // Supported on all targets that support manual MMA
    let tile = tile.with_vector_size::<N>();

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_vectors {
        let elem_idx = i * vector_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
        let offset = row * stride_row + col * stride_col;
        let stage_offset = tile.stage_offset(offset / vector_size as u32);

        fragment[i] = Vector::cast_from(tile.container[stage_offset as usize]);
    }
}

/// This is important to use on CUDA because CUDA's matrices are heavily permuted, being organized
/// into 8x8 chunks with only 32 contiguous bits per thread. `ldmatrix` loads 8 consecutive elements
/// in each thread (if executed with x4), then uses warp shuffles to move the elements to the
/// correct positions for each thread. This currently only supports f16, fp8 needs more handling and
/// packed fp4 isn't supported at all. So these currently fall back to manual loading.
/// tf32 isn't supported by the instruction at all.
#[cube]
fn load_ldmatrix<E: Numeric, N: Size, V: Numeric, NV: Size, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V, NV>,
    fragment: &mut Array<Vector<E, N>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] transposed: bool,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) {
    let stage_vector_size = tile.container.vector_size().comptime();
    let stride = tile.unvectorized_stride();

    let elem_size = E::type_size().comptime();
    let num_regs = def.vectors_per_lane(ident);
    let width = (16 / elem_size / stage_vector_size) as u32;

    let start =
        ldmatrix_offset::<V, A, B, CD>(stride, def, stage_vector_size, ident, layout, tile_size);
    let start = tile.stage_offset(start);

    let row_slice = tile
        .container
        .slice(start as usize, (start + width) as usize);
    let regs = def.load_matrix::<_, N>(&row_slice, ident, num_regs, transposed);

    #[unroll]
    for i in 0..num_regs {
        fragment[i] = Vector::cast_from(regs[i]);
    }
}

/// This logic is horrible and hard to reason about, and very hardcoded. But can't figure out a
/// better way to do it.
#[cube]
pub(crate) fn ldmatrix_offset<E: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    stride: u32,
    def: MmaDefinition<A, B, CD>,
    #[comptime] stage_vector_size: VectorSize,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> u32 {
    let expected_layout = from_cmma_layout(def.vector_layout(ident)).comptime();
    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    let elem_size = E::type_size().comptime();
    let num_regs = def.vectors_per_lane(ident).comptime() as u32;
    let width = 16 / elem_size as u32;
    // Height is always 8, and lanes are divided into blocks of 8.
    let height = 8;

    let (total_rows, total_cols) = match ident {
        MatrixIdent::A => (tile_size.m(), tile_size.k()),
        MatrixIdent::B => (tile_size.k(), tile_size.n()),
        MatrixIdent::Accumulator => (tile_size.m(), tile_size.n()),
    };
    // tile is treated as row-major, if col-major the tile shape is just inverted
    let total_cols = match expected_layout {
        MatrixLayout::RowMajor => total_cols,
        MatrixLayout::ColMajor => total_rows,
    };

    //  Indices are wrapped for < 4 registers.
    let lane = UNIT_POS_PLANE;
    let sub_lane = lane % height;
    let nth_matrix = lane / height % num_regs;

    let tiles_col = total_cols / height;

    // Tiles are arranged in column-major fashion
    let row_offs = (nth_matrix % tiles_col) * 8;
    let col_offs = (nth_matrix / tiles_col) * width;

    let (row, col) = match layout {
        MatrixLayout::RowMajor => (row_offs + sub_lane, col_offs),
        MatrixLayout::ColMajor => (row_offs, col_offs + sub_lane),
    };

    let start = row * stride_row + col * stride_col;
    start / stage_vector_size as u32
}

#[cube]
impl MmaFragmentReader for MmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<
        E: Numeric,
        N: Size,
        V: Numeric,
        NV: Size,
        A: Numeric,
        B: Numeric,
        CD: Numeric,
    >(
        value: &V,
        fragment: &mut Array<Vector<E, N>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] _layout: MatrixLayout,
        #[comptime] _tile_size: TileSize,
        #[comptime] _config: MmaIOConfig,
    ) {
        let num_vectors = def.vectors_per_lane(ident);
        let value = Vector::<E, N>::cast_from(*value);

        #[unroll]
        for i in 0..num_vectors {
            fragment[i] = value;
        }
    }
}

#[cube]
impl<Inner: TileKind> MmaFragmentReader for MmaStageReader<Option<Inner>>
where
    MmaStageReader<Inner>: MmaFragmentReader<TileKind = Inner>,
{
    type TileKind = Option<Inner>;

    fn load_fragment<
        E: Numeric,
        N: Size,
        V: Numeric,
        NV: Size,
        A: Numeric,
        B: Numeric,
        CD: Numeric,
    >(
        tile: &ComptimeOption<Inner::Tile<V, NV>>,
        fragment: &mut Array<Vector<E, N>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] tile_size: TileSize,
        #[comptime] config: MmaIOConfig,
    ) {
        #[comptime]
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => MmaStageReader::<Inner>::load_fragment(
                tile, fragment, def, ident, layout, tile_size, config,
            ),
            ComptimeOption::None => {
                MmaStageReader::<Filled>::load_fragment::<E, N, V, NV, A, B, CD>(
                    &V::from_int(0),
                    fragment,
                    def,
                    ident,
                    layout,
                    tile_size,
                    config,
                )
            }
        }
    }
}
