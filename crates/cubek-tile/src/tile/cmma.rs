//! The tensor-core backing stores: a single fragment ([`CmmaData`]) and an instance's
//! resident partition of them ([`CmmaPartition`]), plus the fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
    std::tensor::layout::CoordsDyn,
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

/// The [`Region`] one staging step windows: the runtime contraction step `ki` on the
/// `contracted` axis, comptime fragment coordinates `(c0, c1)` on the trailing two axes
/// (`contracted` wins where they coincide), `0` elsewhere.
#[cube]
fn step_region(
    #[comptime] space: Space,
    #[comptime] contracted: Axis,
    ki: usize,
    #[comptime] c0: usize,
    #[comptime] c1: usize,
) -> Region {
    let mut coords = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        let axis = comptime!(space.axis_at(p));
        if comptime!(axis == contracted) {
            coords.push(ki as u32);
        } else if comptime!(p == space.rank() - 2) {
            coords.push(c0 as u32);
        } else if comptime!(p == space.rank() - 1) {
            coords.push(c1 as u32);
        } else {
            coords.push(0u32);
        }
    }
    Region::new(coords, space)
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
    /// Stage one contraction step of this memory operand into a register partition — the
    /// register tier's `smem_like` + `copy_from`. One fragment per grid tile of the operand's
    /// trailing two axes; the `contracted` axis stages a single step, positioned at `ki`.
    /// `m`/`n`/`k` are the whole MMA instruction shape (the alloc needs the full triple
    /// whatever this operand's role).
    pub(crate) fn stage_frags(
        &self,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] contracted: Axis,
        ki: usize,
    ) -> Tile<T> {
        let space = comptime!(self.space.clone());
        let a0 = comptime!(space.axis_at(space.rank() - 2));
        let a1 = comptime!(space.axis_at(space.rank() - 1));
        let t0 = comptime!(if a0 == contracted { 1 } else { space.count(a0) });
        let t1 = comptime!(if a1 == contracted { 1 } else { space.count(a1) });

        let mut frags = Sequence::<CmmaData<T>>::new();
        #[unroll]
        for i in 0..t0 {
            #[unroll]
            for j in 0..t1 {
                let mut frag = CmmaData::<T>::alloc(ident, m, n, k);
                let w = self.at(&step_region(comptime!(space.clone()), contracted, ki, i, j));
                match &w.tile_kind {
                    TileKind::Gmem(s) | TileKind::Smem(s) => frag.load_window(s),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("Tile::stage_frags: the source must be a memory window")
                    }
                }
                frags.push(frag);
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<T> {
                frags,
                m_tiles: t0,
                n_tiles: t1,
            }),
            space: comptime!(space),
        }
    }

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
}
