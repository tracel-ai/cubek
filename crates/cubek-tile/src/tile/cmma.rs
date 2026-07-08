//! The tensor-core backing stores: a single fragment ([`CmmaData`]) and an instance's
//! resident partition of them ([`CmmaPartition`]), plus the fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus its comptime config. `cmma::load` picks
/// load-vs-`load_with_layout` by `ident`, and `store`/`cast` need the layout. The
/// fragment's `m`/`n`/`k` and the slice stride come from the tile's [`Space`].
/// `Clone` duplicates the handle, not the fragment — a clone is the same matrix.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaData<T: Numeric> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

/// An instance's resident accumulator partition: `m_tiles × n_tiles` fragments over the
/// output's trailing two axes, row-major comptime-indexed (`mi · n_tiles + ni`). Mirrors
/// cubek-std's `PartitionTile`. `Clone` duplicates the handles, not the fragments.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaPartition<T: Numeric> {
    pub frags: Sequence<CmmaData<T>>,
    #[cube(comptime)]
    pub m_tiles: usize,
    #[cube(comptime)]
    pub n_tiles: usize,
}

#[cube]
impl<T: Numeric> CmmaPartition<T> {
    /// The `(mi, ni)` fragment (a handle clone). Comptime indices only — fragments cannot
    /// be selected at runtime, which is why the partition walk is the comptime microkernel.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> CmmaData<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }
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

    /// Fill this fragment from `mem`'s *window*: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. The slice starts at the window's origin and
    /// rows step by the store's physical row stride, so a window into a larger stage
    /// loads as well as a whole buffer.
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
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Allocate an uninitialized tensor-core fragment as a `Cmma` tile. `m`/`n`/`k`
    /// are the whole MMA tile, passed in full whatever the role.
    pub fn cmma_fragment(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] space: Space,
    ) -> Tile<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        Tile::<T> {
            tile_kind: TileKind::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
            }),
            space: comptime!(space),
        }
    }

    /// Fill this fragment from `src`'s memory *window* (see [`CmmaData::load_window`]).
    pub(crate) fn cmma_load(&mut self, src: &Tile<T>) {
        match &mut self.tile_kind {
            TileKind::Cmma(d) => match &src.tile_kind {
                TileKind::Gmem(s) | TileKind::Smem(s) => d.load_window(s),
                TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                    panic!("Tile::copy_from: cmma→cmma cast not yet wired")
                }
                TileKind::TmaGmem(_) => {
                    panic!("Tile::copy_from: cmma load straight from a tma source not wired")
                }
            },
            // Unreachable: `copy_from` routes here only when `self` is a fragment.
            TileKind::Gmem(_) => (),
            TileKind::Smem(_) => (),
            TileKind::CmmaPartition(_) => (),
            TileKind::TmaGmem(_) => (),
        }
    }

    /// Drain `src` (a `Cmma` fragment) into this memory tile's *window* (see
    /// [`CmmaData::store_window`]).
    pub(crate) fn cmma_store(&mut self, src: &Tile<T>) {
        match &src.tile_kind {
            TileKind::Cmma(s) => match &mut self.tile_kind {
                TileKind::Gmem(d) | TileKind::Smem(d) => s.store_window(d),
                // Unreachable: `copy_from` routes here only when `self` is memory.
                TileKind::Cmma(_) => (),
                TileKind::CmmaPartition(_) => (),
                TileKind::TmaGmem(_) => (),
            },
            // Unreachable: `copy_from` routes here only when `src` is a fragment.
            TileKind::Gmem(_) => (),
            TileKind::Smem(_) => (),
            TileKind::CmmaPartition(_) => (),
            TileKind::TmaGmem(_) => (),
        }
    }
}
