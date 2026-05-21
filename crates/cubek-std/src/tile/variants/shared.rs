//! `TileKind::SharedTile` — the smem stage slot variant.
//!
//! [`SharedTile`] is the enum payload (vectorization erased from the type).
//! [`StridedTile`] is the typed form readers/writers consume.
//! [`SharedTile::wrap`] / [`SharedTile::view`] are pure retypes between them.

use cubecl::{intrinsic, prelude::*, std::Swizzle};

use crate::MatrixLayout;
use crate::stage::{StageMemoryConfig, as_swizzle_object};
use crate::tile::variants::instruction::{
    cmma::cmma_write_to_shared,
    interleaved::interleaved_write_to_shared,
    mma::{MmaFragment, MmaFragmentExpand, mma_write_to_shared},
    plane_vec::planevec_write_to_shared,
    register::register_write_to_shared,
};
use crate::tile::{Tile, TileKind, TileKindExpand, TileScope};

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
/// Typed form of the smem stage slot. `start`/`end`/`stride` are in vector
/// units.
pub struct StridedTile<ES: Numeric, N: Size> {
    /// Slice containing all data for the stage
    pub container: Box<[Vector<ES, N>]>,
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
        container: &[Vector<ES, N>],
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
            container: unsafe { container.as_boxed_unchecked() },
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
        container: &[Vector<ES, N>],
        start: u32,
        end: u32,
        stride: u32,
        swizzle: Swizzle,
        #[comptime] layout: MatrixLayout,
    ) -> StridedTile<ES, N> {
        StridedTile::<ES, N> {
            container: unsafe { container.as_boxed_unchecked() },
            start,
            end,
            stride,
            swizzle,
            layout,
        }
    }
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N> {
    pub fn unvectorized_stride(&self) -> u32 {
        let stage_vector_size = self.container.vector_size();
        self.stride * stage_vector_size as u32
    }
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N> {
    /// Returns the tile as an offset read-only slice. Should only be used when swizzling is
    /// definitely not applicable.
    pub fn as_slice(&self) -> &[Vector<ES, N>] {
        &self.container[self.start as usize..self.end as usize]
    }

    /// Returns the tile as an offset slice. Should only be used when swizzling is definitely not
    /// applicable.
    pub fn as_slice_mut(&mut self) -> &mut [Vector<ES, N>] {
        &mut self.container[self.start as usize..self.end as usize]
    }
}

#[cube]
impl<ES: Numeric, N: Size> StridedTile<ES, N> {
    /// Returns a specific vector from the tile based on coordinates.
    pub fn get_vector(&self, coor_strided: u32, coor_contiguous: u32) -> Vector<ES, N> {
        let offset = coor_strided * self.stride + coor_contiguous;
        let offset_abs = self.start + offset;
        let type_size = Vector::<ES, N>::type_size();
        let offset_swizzled = self.swizzle.apply(offset_abs, type_size);
        self.container[offset_swizzled as usize]
    }

    pub fn stage_offset(&self, relative_offset: u32) -> u32 {
        let offset = self.start + relative_offset;
        let type_size = Vector::<ES, N>::type_size();
        self.swizzle.apply(offset, type_size)
    }

    #[allow(unused_variables)]
    pub fn with_vector_size<N2: Size>(&self) -> StridedTile<ES, N2> {
        let vector_size = N2::value();
        intrinsic!(|scope| {
            let stage_vector_size = self.container.vector_size();

            if vector_size == self.container.vector_size() {
                return self.__expand_with_stage_vector_size_method(scope);
            }

            let current = stage_vector_size;
            let mut out: StridedTileExpand<ES, N2> =
                self.clone().__expand_with_stage_vector_size_method(scope);

            if current < vector_size {
                let ratio = ((vector_size / current) as u32).into_expand(scope);
                let start = self.start.__expand_div_method(scope, ratio);
                let end = self.end.__expand_div_method(scope, ratio);
                let stride = self.stride.__expand_div_method(scope, ratio);
                out.start = start;
                out.end = end;
                out.stride = stride;
            } else {
                let ratio = ((current / vector_size) as u32).into_expand(scope);
                let start = self.start.__expand_mul_method(scope, ratio);
                let end = self.end.__expand_mul_method(scope, ratio);
                let stride = self.stride.__expand_mul_method(scope, ratio);
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
    unsafe fn with_stage_vector_size<N2: Size>(&self) -> StridedTile<ES, N2> {
        StridedTile::<ES, N2> {
            container: unsafe { self.container.with_vector_size::<N2>().as_boxed_unchecked() },
            start: self.start,
            end: self.end,
            stride: self.stride,
            swizzle: self.swizzle,
            layout: self.layout,
        }
    }
}

/// Payload of [`TileKind::SharedTile`]. Vectorization is erased from the
/// type but kept on the runtime slice; project back with [`view`](Self::view).
#[derive(CubeType, Clone)]
pub struct SharedTile<E: Numeric> {
    pub(crate) container: Box<[E]>,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) stride: u32,
    pub(crate) swizzle: Swizzle,
    #[cube(comptime)]
    pub(crate) layout: MatrixLayout,
}

#[cube]
impl<E: Numeric> SharedTile<E> {
    /// Erase the vectorization from a [`StridedTile`].
    pub fn wrap<V: Size>(tile: StridedTile<E, V>) -> SharedTile<E> {
        let container = unsafe { tile.container.downcast_unchecked::<E>() };
        SharedTile::<E> {
            container: unsafe { container.as_boxed_unchecked() },
            start: tile.start,
            end: tile.end,
            stride: tile.stride,
            swizzle: tile.swizzle,
            layout: tile.layout,
        }
    }

    /// Project back to a typed [`StridedTile`]. Must match the original
    /// vectorization.
    pub fn view<V: Size>(&self) -> StridedTile<E, V> {
        let container = unsafe { self.container.downcast_unchecked::<Vector<E, V>>() };
        StridedTile::<E, V> {
            container: unsafe { container.as_boxed_unchecked() },
            start: self.start,
            end: self.end,
            stride: self.stride,
            swizzle: self.swizzle,
            layout: self.layout,
        }
    }
}

#[cube]
impl<E: Numeric> SharedTile<E> {
    /// Write-back leg of `Tile::copy_from`: routes to the source variant's
    /// `*_write_to_shared` helper.
    pub fn copy_from<SE: Numeric, SS: Size, L: Numeric, R: Numeric, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
    ) {
        match &source.kind {
            TileKind::Cmma(t) => cmma_write_to_shared::<E, SS, SE>(self, &t.matrix),
            TileKind::Bounce(b) => cmma_write_to_shared::<E, SS, SE>(self, &b.cmma.matrix),
            TileKind::Mma(t) => match &t.fragment {
                MmaFragment::Acc(f) => {
                    mma_write_to_shared::<E, SS, SE, L, R>(self, f, t.tile_size, t.mma_io_config);
                }
                MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                    panic!("Mma write_to_shared only supported for Acc role")
                }
            },
            TileKind::Register(t) => {
                register_write_to_shared::<E, SS, SE>(self, &t.tile.data, t.tile_size);
            }
            TileKind::PlaneVec(t) => {
                planevec_write_to_shared::<SE, E, SS>(
                    self,
                    &t.data,
                    t.tile_size,
                    t.reduce_vector_size,
                );
            }
            TileKind::Interleaved(t) => {
                interleaved_write_to_shared::<E, SS, SE>(self, &t.data, t.tile_size);
            }
            _ => panic!("SharedTile::copy_from: unsupported source variant"),
        }
    }
}
