use cubecl::{prelude::*, std::Swizzle};

use crate::{
    MatrixLayout,
    stage::{
        StageMemoryConfig, StridedStageMemory, TilingLayout, TilingLayoutEnum, TilingOrderEnum,
    },
    tile::SharedTile,
};
use cubecl::std::tensor::layout::Coords2d;

/// Payload of [`TileKind::Stage`](crate::tile::TileKind): non-owning view of a
/// [`StridedStageMemory`] buffer, with vector size and tiling layout erased
/// from the type and held as comptime metadata.
#[derive(CubeType, Clone)]
pub struct StageTile<E: Numeric> {
    pub smem: Box<[E]>,
    pub swizzle: Swizzle,
    #[cube(comptime)]
    pub config: StageMemoryConfig,
    #[cube(comptime)]
    pub tiling_layout: TilingLayoutEnum,
}

#[cube]
impl<E: Numeric> StageTile<E> {
    pub fn wrap<NS: Size, T: TilingLayout>(stage: &StridedStageMemory<E, NS, T>) -> StageTile<E> {
        let typed = stage.as_slice::<NS>();
        let erased: &[E] = unsafe { typed.downcast_unchecked::<E>() };
        StageTile::<E> {
            smem: unsafe { erased.as_boxed_unchecked() },
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}

#[cube]
impl<E: Numeric> StageTile<E> {
    pub fn wrap_mut<NS: Size, T: TilingLayout>(
        stage: &mut StridedStageMemory<E, NS, T>,
    ) -> StageTile<E> {
        let typed = stage.as_slice_mut::<NS>();
        let erased = unsafe { typed.downcast_mut_unchecked::<E>() };
        StageTile::<E> {
            smem: unsafe { erased.as_boxed_unchecked() },
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}

#[cube]
impl<E: Numeric> StageTile<E> {
    /// [`SharedTile`] view of the tile at `coord`. Dispatches on the
    /// comptime [`TilingLayoutEnum`].
    pub fn get_tile(&self, coord: Coords2d) -> SharedTile<E> {
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

                        SharedTile::<E> {
                            container: self.smem.clone(),
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

                        SharedTile::<E> {
                            container: self.smem.clone(),
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
                let nth =
                    to_nth_tile_contiguous(order, coord, tile_count_x, tile_count_y, self.config);

                let length = comptime!(self.config.elements_per_tile() / stage_vector_size);
                let stride_elements = match matrix_layout {
                    MatrixLayout::RowMajor => comptime!(self.config.elements_per_tile_along_col),
                    MatrixLayout::ColMajor => comptime!(self.config.elements_per_tile_along_row),
                };
                let stride = comptime!(stride_elements / stage_vector_size);
                let start = (comptime!(self.config.elements_per_tile()) * nth) / stage_vector_size;

                SharedTile::<E> {
                    container: self.smem.clone(),
                    start,
                    end: start + length,
                    stride,
                    swizzle: self.swizzle,
                    layout: matrix_layout,
                }
            }
            TilingLayoutEnum::Other => {
                panic!("StageTile::get_tile: TilingLayoutEnum::Other (Tma/None) not supported")
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
    #[comptime] config: StageMemoryConfig,
) -> u32 {
    let (row, col) = coord;
    match order {
        TilingOrderEnum::RowMajor => row * tile_count_cols + col,
        TilingOrderEnum::ColMajor => col * tile_count_rows + row,
        TilingOrderEnum::Ordered => {
            let group_rows = comptime!(tile_count_rows / config.num_planes);
            let tiles_per_group = comptime!(group_rows * tile_count_cols);
            let group = row / group_rows;
            let local_row = row % group_rows;
            let pos_within_group = col * group_rows + local_row;
            group * tiles_per_group + pos_within_group
        }
        TilingOrderEnum::Tma => {
            panic!("StageTile::get_tile: Tma tiling not yet supported on type-erased view")
        }
    }
}
