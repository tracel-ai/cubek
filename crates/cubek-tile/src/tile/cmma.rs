//! The tensor-core encoding of a plane tile ([`CmmaData`]) and its fragment↔memory transports.
//! The grid over it is the encoding-blind [`PlanePartition`](super::plane).

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment: a clone is the same matrix.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaData<T: Numeric> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[cube]
impl<T: Numeric> CmmaData<T> {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, passed in
    /// full whatever the role; the layout is `RowMajor` (how the stages are laid out).
    pub(crate) fn alloc(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
    ) -> CmmaData<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, MatrixLayout::RowMajor) };
        CmmaData::<T> {
            matrix,
            ident,
            layout: MatrixLayout::RowMajor,
        }
    }

    /// An uninitialized fragment presented as a `Cmma` tile. `m`/`n`/`k` are the whole
    /// MMA tile, passed in full whatever the role.
    pub fn fragment(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] space: Space,
    ) -> Tile<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        Tile::<T> {
            tile_kind: TileKind::new_PlaneTile(PlaneTile::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
            })),
            space: comptime!(space),
        }
    }

    /// Zero the fragment.
    pub(crate) fn zero(&mut self) {
        cmma::fill(&mut self.matrix, T::from_int(0));
    }

    /// Fill this fragment from `mem`'s *window*: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. Rows step by the store's physical row
    /// stride, so a window into a larger stage loads like a whole buffer.
    pub(crate) fn load_window(&mut self, mem: &MemData<T>) {
        let stride = mem.row_stride();
        match comptime!(self.ident) {
            MatrixIdent::Accumulator => cmma::load_with_layout(
                &mut self.matrix,
                mem.window_slice(),
                stride,
                comptime!(self.layout),
            ),
            _ => cmma::load(&mut self.matrix, mem.window_slice(), stride),
        }
    }

    /// Drain this fragment into `mem`'s *window* (origin offset, physical row stride).
    pub(crate) fn store_window(&self, mem: &mut MemData<T>) {
        let stride = mem.row_stride();
        cmma::store(
            mem.window_slice_mut(),
            &self.matrix,
            stride,
            comptime!(self.layout),
        )
    }

    /// Drain this fragment into `mem`'s *window*, casting `T` down to the sink's element
    /// type first: a register accumulator (e.g. `f32`) is wider than the stored output
    /// (e.g. `f16`). The cast is a no-op when the types match.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let stride = mem.row_stride();
        let casted: Matrix<Out> = cmma::cast(&self.matrix);
        cmma::store(
            mem.window_slice_mut(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}
