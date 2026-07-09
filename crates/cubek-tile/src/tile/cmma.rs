//! The tensor-core backing stores: a single fragment ([`CmmaData`]) and a partition of
//! them ([`CmmaPartition`]), plus the fragment↔memory transports. Everything that knows
//! how a space maps to a fragment grid (its shape, its windows, its classification) lives
//! here, on the kind — [`Tile`] stays structure.

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

/// A partition of cmma fragments: `m_tiles × n_tiles` over the tile's trailing two axes,
/// row-major comptime-indexed (`mi · n_tiles + ni`). Mirrors cubek-std's `PartitionTile`.
/// `Clone` duplicates the handles, not the fragments.
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
    /// be selected at runtime, which is why the register tier walks statically.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> CmmaData<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The tile whose partition mirrors `space`'s fragment grid, accumulator fragments
    /// uninitialized — the register form of an accumulator over `space`, `smem_like`'s
    /// register sibling. `k` is the contraction depth of the instruction (the space's
    /// own axes only give `m`/`n`).
    pub(crate) fn mirror(#[comptime] space: Space, #[comptime] k: usize) -> Tile<T> {
        let (m_tiles, n_tiles) = comptime!(partition_shape(&space));
        let fin = comptime!(space.final_space());
        let m = comptime!(fin.extent_at(fin.rank() - 2));
        let n = comptime!(fin.extent_at(fin.rank() - 1));

        let mut frags = Sequence::<CmmaData<T>>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                frags.push(CmmaData::<T>::alloc(MatrixIdent::Accumulator, m, n, k));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<T> {
                frags,
                m_tiles,
                n_tiles,
            }),
            space: comptime!(space),
        }
    }

    /// One contraction step of the memory `operand` staged into a partition tile — the
    /// register tier's stage fill. One fragment per grid tile of the operand's trailing
    /// two axes; the `contracted` axis stages a single step, positioned at `ki`.
    /// `m`/`n`/`k` are the whole MMA instruction shape (the alloc needs the full triple
    /// whatever the operand's role).
    pub(crate) fn stage(
        operand: &Tile<T>,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] contracted: Axis,
        ki: usize,
    ) -> Tile<T> {
        let space = comptime!(operand.space.clone());
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
                let w = operand.at(&step_region(comptime!(space.clone()), contracted, ki, i, j));
                match &w.tile_kind {
                    TileKind::Gmem(s) | TileKind::Smem(s) => frag.load_window(s),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::stage: the source must be a memory window")
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

    /// Fill each fragment from its final window of `src` — the transport behind
    /// [`copy_from`](Tile::copy_from)'s partition-destination pairing. Iterates the grid
    /// in the partition's own row-major fragment order.
    pub(crate) fn fill_from(&self, src: &Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.fragment_window(mi, ni);
                match &window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.load_window(g),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::fill_from: the source must be memory")
                    }
                }
            }
        }
    }

    /// Drain each fragment into its final window of `dst` — [`fill_from`](Self::fill_from)'s
    /// inverse.
    pub(crate) fn drain_into(&self, dst: &mut Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni);
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::drain_into: the sink must be memory")
                    }
                }
            }
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Descend to the `(mi, ni)` fragment's final window — pure structure (regions and
    /// levels), no kind knowledge: an instance level hands this instance a single region
    /// (`region(0)`, hardware position folded in); the partition level takes the static
    /// region at the partition coordinates.
    fn fragment_window(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> Tile<T> {
        let space = comptime!(self.space.clone());
        let sub = match comptime!(partition_level(&space)) {
            None => {
                let walk = Walk::over(self.runtime_space());
                self.at(&walk.region(0))
            }
            Some(_) => self.at_static(comptime!(StaticRegion::trailing(&space, mi, ni))),
        };
        match comptime!(sub.space.partitioner()) {
            Partitioner::Final(_) => sub,
            Partitioner::Level(_) => sub.fragment_window(mi, ni),
        }
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

/// The per-instance tile count of `axis` at this level, `None` when it is runtime.
fn per_instance_tiles(level: &Space, axis: Axis) -> Option<usize> {
    let edge = level.partitioner().edge(axis);
    match level.partitioner().distribution(axis) {
        Distribution::Sequential => match level.extent_raw(axis) {
            Extent::Static(e) => Some(e.div_ceil(edge)),
            Extent::Dynamic => None,
        },
        Distribution::Spatial { coverage, .. } => match coverage {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(n) => match level.extent_raw(axis) {
                Extent::Static(e) => Some(e.div_ceil(edge).div_ceil(n)),
                Extent::Dynamic => None,
            },
        },
    }
}

/// Classify the current level of a space that backs fragments: `None` for an *instance*
/// level (one tile per axis), or the trailing-two-axes tile counts for the *partition*
/// level (comptime sequential tiles this instance owns wholesale). Anything else cannot
/// back fragments and panics at comptime.
pub(crate) fn partition_level(space: &Space) -> Option<(usize, usize)> {
    if space.is_final() {
        return None;
    }
    if space
        .axes()
        .all(|a| per_instance_tiles(space, a) == Some(1))
    {
        return None;
    }
    let rank = space.rank();
    for (p, axis) in space.axes().enumerate() {
        assert!(
            matches!(
                space.partitioner().distribution(axis),
                Distribution::Sequential
            ),
            "cmma partition level: every axis below the instance split must be sequential"
        );
        let tiles = per_instance_tiles(space, axis)
            .expect("cmma partition level: tile counts must be comptime");
        assert!(
            p >= rank - 2 || tiles == 1,
            "cmma partition level: leading (batch) axes must hand out one tile"
        );
    }
    Some((
        per_instance_tiles(space, space.axis_at(rank - 2)).unwrap(),
        per_instance_tiles(space, space.axis_at(rank - 1)).unwrap(),
    ))
}

/// The whole remaining walk's fragment grid for one instance: `(1, 1)` when every level
/// is an instance level, else the single partition level's tile counts.
fn partition_shape(space: &Space) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut level = space.clone();
    while !level.is_final() {
        if let Some(counts) = partition_level(&level) {
            assert!(
                shape == (1, 1),
                "cmma accumulator: at most one partition level"
            );
            shape = counts;
        }
        level = level.divide();
    }
    shape
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
            tile_kind: TileKind::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
            }),
            space: comptime!(space),
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
