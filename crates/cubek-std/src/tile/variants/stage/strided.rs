use cubecl::{prelude::*, std::Swizzle};

use crate::{
    MatrixLayout,
    stage::StageMemoryConfig,
    tile::{
        SharedTile,
        variants::stage::{
            TilingLayout, TilingLayoutEnum, TilingOrderEnum, memory::StridedStageMemory,
        },
    },
};
use cubecl::std::tensor::layout::Coords2d;

/// Type-erased view of a [`StridedStageMemory`] buffer for use as a tile kind.
///
/// Mirrors [`SharedTile`](crate::tile::variants::SharedTile)'s pattern: the
/// vector-size (`NS`) and tiling-layout (`T`) generics that parameterize the
/// underlying allocation are dropped at the type level. The vector size lives
/// in `config.vector_size`, the tiling layout is encoded as a comptime
/// [`TilingLayoutEnum`], and the smem slice is held as a scalar `Slice<E, IO>`
/// (re-typed via `downcast_unchecked` at construction).
///
/// Lifecycle: the underlying `StridedStageMemory` continues to own the
/// allocation; `StridedStage` is a non-owning view installed as a
/// [`TileKind::Stage`](crate::tile::TileKind) payload.
#[derive(CubeType, Clone, Copy)]
pub struct StridedStage<E: Numeric, IO: SliceVisibility = ReadOnly> {
    /// Scalar-typed slice covering the active buffer of the underlying smem.
    /// Re-typed back to `Slice<Vector<E, NS>, IO>` at lookup time via
    /// `with_vector_size::<NS>()` using `config.vector_size`.
    pub smem: Slice<E, IO>,
    pub swizzle: Swizzle,
    #[cube(comptime)]
    pub config: StageMemoryConfig,
    #[cube(comptime)]
    pub tiling_layout: TilingLayoutEnum,
}

#[cube]
impl<E: Numeric> StridedStage<E, ReadOnly> {
    /// Wraps a `StridedStageMemory`'s active buffer as a type-erased view.
    /// `NS` and `T` are consumed here and become runtime/comptime metadata.
    pub fn wrap<NS: Size, T: TilingLayout>(
        stage: &StridedStageMemory<E, NS, T>,
    ) -> StridedStage<E, ReadOnly> {
        let typed = stage.as_slice::<NS>();
        let erased: Slice<E, ReadOnly> = unsafe { typed.downcast_unchecked::<E>() };
        StridedStage::<E, ReadOnly> {
            smem: erased,
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}

#[cube]
impl<E: Numeric> StridedStage<E, ReadWrite> {
    /// Mutable variant of [`StridedStage::wrap`].
    pub fn wrap_mut<NS: Size, T: TilingLayout>(
        stage: &mut StridedStageMemory<E, NS, T>,
    ) -> StridedStage<E, ReadWrite> {
        let typed = stage.as_slice_mut::<NS>();
        let erased: SliceMut<E> = unsafe { typed.downcast_unchecked::<E>() };
        StridedStage::<E, ReadWrite> {
            smem: erased,
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}

#[cube]
impl<E: Numeric, IO: SliceVisibility> StridedStage<E, IO> {
    /// Returns a [`SharedTile`] view of the tile at `coord` in the stage.
    ///
    /// Dispatches on the comptime [`TilingLayoutEnum`]; mirrors today's
    /// [`TilingLayout::get_tile`] but works against the type-erased view.
    /// `start`/`end`/`stride` in the returned `SharedTile` are in vector
    /// units (using `self.config.vector_size`); the wrapped slice is
    /// scalar-typed and is re-cast at downstream `.view::<NS>()` sites.
    pub fn get_tile(&self, coord: Coords2d) -> SharedTile<E, IO> {
        let (row, col) = coord;
        let stage_vector_size = comptime!(self.config.vector_size);
        let matrix_layout = comptime!(self.config.matrix_layout);

        match comptime!(self.tiling_layout) {
            TilingLayoutEnum::Strided => {
                let tile_count_x = comptime!(self.config.tiles_per_stage_along_row());
                let tile_count_y = comptime!(self.config.tiles_per_stage_along_col());

                match matrix_layout {
                    MatrixLayout::RowMajor => {
                        let tile_size_x = comptime!(self.config.elements_per_tile_along_row);
                        let tile_size_y =
                            comptime!(self.config.elements_per_tile_along_col / stage_vector_size);

                        let stride = comptime!(tile_count_y * tile_size_y);
                        let length = comptime!((tile_size_x - 1) * stride + tile_size_y);
                        let start = row * tile_size_x * stride + col * tile_size_y;

                        SharedTile::<E, IO> {
                            container: self.smem,
                            start,
                            end: start + length,
                            stride,
                            swizzle: self.swizzle,
                            layout: matrix_layout,
                        }
                    }
                    MatrixLayout::ColMajor => {
                        let tile_size_x =
                            comptime!(self.config.elements_per_tile_along_row / stage_vector_size);
                        let tile_size_y = comptime!(self.config.elements_per_tile_along_col);

                        let stride = comptime!(tile_count_x * tile_size_x);
                        let length = comptime!((tile_size_y - 1) * stride + tile_size_x);
                        let start = row * tile_size_x + col * tile_size_y * stride;

                        SharedTile::<E, IO> {
                            container: self.smem,
                            start,
                            end: start + length,
                            stride,
                            swizzle: self.swizzle,
                            layout: matrix_layout,
                        }
                    }
                }
            }
            TilingLayoutEnum::Contiguous(order) => {
                let tile_count_x = comptime!(self.config.tiles_per_stage_along_row());
                let tile_count_y = comptime!(self.config.tiles_per_stage_along_col());
                let nth = to_nth_tile_contiguous(order, coord, tile_count_x, tile_count_y);

                let length = comptime!(self.config.elements_per_tile() / stage_vector_size);
                let stride_elements = match matrix_layout {
                    MatrixLayout::RowMajor => comptime!(self.config.elements_per_tile_along_col),
                    MatrixLayout::ColMajor => comptime!(self.config.elements_per_tile_along_row),
                };
                let stride = comptime!(stride_elements / stage_vector_size);
                let start = (comptime!(self.config.elements_per_tile()) * nth) / stage_vector_size;

                SharedTile::<E, IO> {
                    container: self.smem,
                    start,
                    end: start + length,
                    stride,
                    swizzle: self.swizzle,
                    layout: matrix_layout,
                }
            }
            TilingLayoutEnum::Other => {
                panic!("StridedStage::get_tile: TilingLayoutEnum::Other (Tma/None) not supported")
            }
        }
    }
}

#[cube]
fn to_nth_tile_contiguous(
    #[comptime] order: TilingOrderEnum,
    coord: Coords2d,
    #[comptime] tile_count_rows: u32,
    #[comptime] tile_count_cols: u32,
) -> u32 {
    let (row, col) = coord;
    match order {
        TilingOrderEnum::RowMajor => row * tile_count_cols + col,
        TilingOrderEnum::ColMajor => col * tile_count_rows + row,
        TilingOrderEnum::Ordered => {
            panic!("StridedStage::get_tile: Ordered tiling not yet supported on type-erased view")
        }
        TilingOrderEnum::Tma => {
            panic!("StridedStage::get_tile: Tma tiling not yet supported on type-erased view")
        }
    }
}
