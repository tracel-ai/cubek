use cubecl::intrinsic;
use cubecl::prelude::*;
use cubecl::std::Swizzle;

use crate::MatrixLayout;
use crate::stage::StageMemoryConfig;
use crate::stage::as_swizzle_object;

#[derive(CubeType, Clone, Copy)]
/// Tile with a linear major dimension, and a strided minor dimension.
/// Basic tile kind supported by all stage matmuls.
pub struct StridedTile<ES: Numeric, N: Size, IO: SliceVisibility = ReadOnly> {
    /// Slice containing all data for the stage
    pub stage: Slice<Vector<ES, N>, IO>,
    /// Offset of the tile in the stage
    pub start: u32,
    /// End of the tile in the stage, may be wrong with swizzle
    pub end: u32,
    /// Stride between each row/col, depending on MatrixLayout (the other is assumed to be 1)
    pub stride: u32,
    /// Swizzle object to transform the index
    pub swizzle: Swizzle,
    #[cube(comptime)]
    /// Layout of the tile (row-major or column-major).
    pub layout: MatrixLayout,
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N> {
    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous(
        stage: Slice<Vector<ES, N>>,
        start: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, N> {
        let len = config.elements_per_tile() / config.vector_size;
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_per_tile_along_col,
            MatrixLayout::ColMajor => config.elements_per_tile_along_row,
        };

        let stride = stride / config.vector_size;

        StridedTile::<ES, N> {
            stage,
            start,
            end: start + len,
            stride,
            swizzle: as_swizzle_object(config.swizzle),
            layout,
        }
    }

    /// Creates a tile from a contiguous slice of data.
    ///
    /// The slice length must exactly match the tile size.
    pub fn new_contiguous_mut(
        stage: Slice<Vector<ES, N>, ReadWrite>,
        start: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, N, ReadWrite> {
        let len = config.elements_per_tile() / config.vector_size;
        let layout = config.matrix_layout;
        let stride = match layout {
            MatrixLayout::RowMajor => config.elements_per_tile_along_col,
            MatrixLayout::ColMajor => config.elements_per_tile_along_row,
        };

        let stride = stride / config.vector_size;

        StridedTile::<ES, N, ReadWrite> {
            stage,
            start,
            end: start + len,
            stride,
            swizzle: as_swizzle_object(config.swizzle),
            layout,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
    pub fn new_strided(
        stage: Slice<Vector<ES, N>>,
        start: u32,
        end: u32,
        stride: u32,
        swizzle: Swizzle,
        #[comptime] layout: MatrixLayout,
    ) -> StridedTile<ES, N> {
        StridedTile::<ES, N> {
            stage,
            start,
            end,
            stride,
            swizzle,
            layout,
        }
    }

    /// Creates a tile from a strided slice of data.
    ///
    /// The slice must include all elements of the tile, though it may include unused gaps.
    pub fn new_strided_mut(
        stage: Slice<Vector<ES, N>, ReadWrite>,
        start: u32,
        end: u32,
        stride: u32,
        swizzle: Swizzle,
        #[comptime] layout: MatrixLayout,
    ) -> StridedTile<ES, N, ReadWrite> {
        StridedTile::<ES, N, ReadWrite> {
            stage,
            start,
            end,
            stride,
            swizzle,
            layout,
        }
    }
}

#[cube]
impl<ES: Numeric, N: Size, IO: SliceVisibility> StridedTile<ES, N, IO> {
    pub fn unvectorized_stride(&self) -> u32 {
        let stage_vector_size = self.stage.vector_size();
        self.stride * stage_vector_size as u32
    }
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N, ReadOnly> {
    /// Returns the tile as an offset slice. Should only be used when swizzling is definitely not
    /// applicable.
    pub fn as_slice(&self) -> Slice<Vector<ES, N>, ReadOnly> {
        self.stage.slice(self.start as usize, self.end as usize)
    }
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N, ReadWrite> {
    /// Returns the tile as an offset slice. Should only be used when swizzling is definitely not
    /// applicable.
    pub fn as_slice_mut(&self) -> Slice<Vector<ES, N>, ReadWrite> {
        self.stage
            .slice(self.start as usize, self.end as usize)
            .as_mut_unchecked()
    }
}

#[cube]
impl<ES: Numeric, N: Size, IO: SliceVisibility> StridedTile<ES, N, IO> {
    /// Returns a specific vector from the tile based on coordinates.
    pub fn get_vector(&self, coor_strided: u32, coor_contiguous: u32) -> Vector<ES, N> {
        let offset = coor_strided * self.stride + coor_contiguous;
        let offset_abs = self.start + offset;
        let type_size = Vector::<ES, N>::type_size();
        let offset_swizzled = self.swizzle.apply(offset_abs, type_size);
        self.stage[offset_swizzled as usize]
    }

    pub fn stage_offset(&self, relative_offset: u32) -> u32 {
        let offset = self.start + relative_offset;
        let type_size = Vector::<ES, N>::type_size();
        self.swizzle.apply(offset, type_size)
    }

    #[allow(unused_variables)]
    pub fn with_vector_size<N2: Size>(&self) -> StridedTile<ES, N2, IO> {
        let vector_size = N2::value();
        intrinsic!(|scope| {
            let stage_vector_size = self.stage.vector_size();

            if vector_size == self.stage.vector_size() {
                return self.__expand_with_stage_vector_size_method(scope);
            }

            let current = stage_vector_size;
            let mut out: StridedTileExpand<ES, N2, IO> =
                self.clone().__expand_with_stage_vector_size_method(scope);

            if current < vector_size {
                let ratio = (vector_size / current) as u32;
                let end = cubecl::frontend::div::expand(scope, self.end, ratio.into());
                let start = cubecl::frontend::div::expand(scope, self.start, ratio.into());
                let stride =
                    cubecl::frontend::div::expand(scope, self.stride, (ratio as u32).into());
                out.start = start;
                out.end = end;
                out.stride = stride;
            } else {
                let ratio = (current / vector_size) as u32;
                let start = cubecl::frontend::mul::expand(scope, self.start, ratio.into());
                let end = cubecl::frontend::mul::expand(scope, self.end, ratio.into());
                let stride = cubecl::frontend::mul::expand(scope, self.stride, ratio.into());
                out.start = start;
                out.end = end;
                out.stride = stride;
            }

            out
        })
    }

    /// Cast only the stage vector size. This leaves the tile in an invalid state - start, end and
    /// stride must be adjusted accordingly.
    /// # Safety
    /// Must not be used without further metadata adjustments
    #[allow(unused)]
    unsafe fn with_stage_vector_size<N2: Size>(self) -> StridedTile<ES, N2, IO> {
        StridedTile::<ES, N2, IO> {
            stage: self.stage.with_vector_size::<N2>(),
            start: self.start,
            end: self.end,
            stride: self.stride,
            swizzle: self.swizzle,
            layout: self.layout,
        }
    }
}
