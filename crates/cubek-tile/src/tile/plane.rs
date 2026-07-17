//! The plane-level tile ([`PlaneTile`]) and the grid of them one plane owns
//! ([`PlanePartition`]).
//!
//! A plane tile is owned by a plane and sliced across its lanes, never unit-addressable: cmma's
//! `Matrix` is `MatrixScope::Plane`, and manual mma's registers index by `UNIT_POS_PLANE`. The two
//! are one concept with two encodings ([`CmmaData`], [`MmaData`]), so the partition over them is
//! encoding-blind and written once. Cube-level MMA is a different scope and does not belong here.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// One plane-level tile, by encoding. The [`Leaf`] picks which.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum PlaneTile<T: Numeric> {
    Cmma(CmmaData<T>),
    Mma(MmaData<T>),
}

#[cube]
impl<T: Numeric> PlaneTile<T> {
    /// An accumulator tile over the whole `m × n` MMA tile, uninitialized.
    pub(crate) fn acc(
        #[comptime] leaf: Leaf,
        #[comptime] m: usize,
        #[comptime] n: usize,
    ) -> PlaneTile<T> {
        match comptime!(leaf) {
            Leaf::Cmma { k } => {
                PlaneTile::new_Cmma(CmmaData::<T>::alloc(MatrixIdent::Accumulator, m, n, k))
            }
            Leaf::Mma { k, io } => {
                PlaneTile::new_Mma(MmaData::<T>::acc(m, n, k, MatrixLayout::RowMajor, io))
            }
            Leaf::Register => {
                panic!("Tile::promote: the register leaf runs in place — nothing to promote")
            }
        }
    }

    /// An operand tile in role `ident`, uninitialized. `k` is the operand's own contraction
    /// depth, not the leaf's.
    pub(crate) fn operand(
        #[comptime] leaf: Leaf,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
    ) -> PlaneTile<T> {
        match comptime!(leaf) {
            Leaf::Cmma { .. } => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k)),
            Leaf::Mma { io, .. } => match comptime!(ident) {
                MatrixIdent::A => {
                    PlaneTile::new_Mma(MmaData::<T>::lhs(m, n, k, MatrixLayout::RowMajor, io))
                }
                MatrixIdent::B => {
                    PlaneTile::new_Mma(MmaData::<T>::rhs(m, n, k, MatrixLayout::RowMajor, io))
                }
                MatrixIdent::Accumulator => {
                    panic!("PlaneTile::operand: an accumulator is not an operand")
                }
            },
            Leaf::Register => panic!("PlaneTile::operand: the register leaf has no plane tile"),
        }
    }

    pub(crate) fn zero(&mut self) {
        match self {
            PlaneTile::Cmma(d) => d.zero(),
            PlaneTile::Mma(d) => d.zero(),
        }
    }

    pub(crate) fn load_window(&mut self, mem: &MemData<T>) {
        match self {
            PlaneTile::Cmma(d) => d.load_window(mem),
            PlaneTile::Mma(d) => d.load_window(mem),
        }
    }

    pub(crate) fn store_window(&self, mem: &mut MemData<T>) {
        match self {
            PlaneTile::Cmma(d) => d.store_window(mem),
            PlaneTile::Mma(d) => d.store_window(mem),
        }
    }

    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        match self {
            PlaneTile::Cmma(d) => d.store_cast_window(mem),
            PlaneTile::Mma(d) => d.store_cast_window(mem),
        }
    }
}

/// The grid of plane tiles one plane owns: `m_tiles × n_tiles` over the tile's trailing two axes,
/// row-major comptime-indexed (`mi · n_tiles + ni`). Blind to the tiles' encoding.
/// `Clone` duplicates the handles, not the tiles.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct PlanePartition<T: Numeric> {
    pub frags: Sequence<PlaneTile<T>>,
    #[cube(comptime)]
    pub m_tiles: usize,
    #[cube(comptime)]
    pub n_tiles: usize,
}

#[cube]
impl<T: Numeric> PlanePartition<T> {
    /// The `(mi, ni)` tile (a handle clone). Comptime indices only: plane tiles cannot be
    /// selected at runtime.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> PlaneTile<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The `m_tiles × n_tiles` sub-partition at `(mi, ni)` (handle clones, so its tiles are the
    /// parent's): a stacked partition level selects a block where the grid itself selects one.
    pub(crate) fn window(
        &self,
        #[comptime] mi: usize,
        #[comptime] ni: usize,
        #[comptime] m_tiles: usize,
        #[comptime] n_tiles: usize,
    ) -> PlanePartition<T> {
        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for i in 0..m_tiles {
            #[unroll]
            for j in 0..n_tiles {
                frags.push(self.at(comptime!(mi + i), comptime!(ni + j)));
            }
        }
        PlanePartition::<T> {
            frags,
            m_tiles,
            n_tiles,
        }
    }

    /// The plane-resident form of an accumulator over `space`: a partition mirroring its grid,
    /// tiles uninitialized. `promote` is purely structural; the caller states the init.
    pub(crate) fn mirror(#[comptime] space: Space) -> Tile<T> {
        let leaf = comptime!(space.partitioner().leaf());
        let (m_tiles, n_tiles) = comptime!(partition_shape(&space));
        let fin = comptime!(space.final_space());
        let m = comptime!(fin.extent_at(fin.rank() - 2));
        let n = comptime!(fin.extent_at(fin.rank() - 1));

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                frags.push(PlaneTile::<T>::acc(leaf, m, n));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles,
                n_tiles,
            }),
            space: comptime!(space),
        }
    }

    /// The staging store for one region of an operand under `out`'s contraction: a partition
    /// mirroring the region's grid, tiles uninitialized; [`copy_from`](Tile::copy_from) fills it.
    pub(crate) fn store(#[comptime] window: Space, #[comptime] out: Space) -> Tile<T> {
        let leaf = comptime!(out.partitioner().leaf());
        let a0 = comptime!(window.axis_at(window.rank() - 2));
        let a1 = comptime!(window.axis_at(window.rank() - 1));
        let t0 = comptime!(window.count(a0));
        let t1 = comptime!(window.count(a1));

        // `A` is `m×k`, `B` is `k×n`: the operand's role is where its contracted axis sits.
        let contracted = comptime!(window.contraction(&out));
        let ident = comptime!(if contracted == a1 {
            MatrixIdent::A
        } else {
            assert!(
                contracted == a0,
                "PlanePartition::store: the contracted axis must be one of the trailing two"
            );
            MatrixIdent::B
        });
        let out_fin = comptime!(out.final_space());
        let m = comptime!(out_fin.extent_at(out_fin.rank() - 2));
        let n = comptime!(out_fin.extent_at(out_fin.rank() - 1));
        let k = comptime!(window.final_space().extent(contracted));

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _i in 0..t0 {
            #[unroll]
            for _j in 0..t1 {
                frags.push(PlaneTile::<T>::operand(leaf, ident, m, n, k));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles: t0,
                n_tiles: t1,
            }),
            space: comptime!(window),
        }
    }

    /// Fill each tile from its final window of `src`, in the partition's row-major order.
    pub(crate) fn fill_from(&self, src: &Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.fragment_window(mi, ni);
                match &window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.load_window(g),
                    TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                        panic!("PlanePartition::fill_from: the source must be memory")
                    }
                }
            }
        }
    }

    /// Zero every tile.
    pub(crate) fn zero(&self) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                frag.zero();
            }
        }
    }

    /// Drain each tile into its final window of `dst`; [`fill_from`](Self::fill_from)'s inverse.
    pub(crate) fn drain_into(&self, dst: &mut Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni);
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g),
                    TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                        panic!("PlanePartition::drain_into: the sink must be memory")
                    }
                }
            }
        }
    }

    /// Drain each tile into its final window of `dst`, casting `T` to `dst`'s element type first:
    /// a plane accumulator (e.g. `f32`) written to a narrower output (e.g. `f16`).
    pub(crate) fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni);
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_cast_window(g),
                    TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                        panic!("PlanePartition::drain_cast_into: the sink must be memory")
                    }
                }
            }
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Descend to the `(mi, ni)` tile's final window: an instance level hands this instance a
    /// single region; a partition level takes its own digit of the grid coordinates — the grid may
    /// be split across stacked levels, so each consumes the high digits (the levels below it are
    /// the place value) and passes the rest down.
    pub(crate) fn fragment_window(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> Tile<T> {
        let space = comptime!(self.space.clone());
        match comptime!(partition_level(&space)) {
            None => {
                let walk = Walk::over(self.runtime_space());
                let sub = self.at(&walk.region(0));
                match comptime!(sub.space.partitioner()) {
                    Partitioner::Final(_) => sub,
                    Partitioner::Level(_) => sub.fragment_window(mi, ni),
                }
            }
            Some(_) => {
                let (bm, bn) = comptime!(partition_shape(&space.divide()));
                let region = Region::trailing(
                    comptime!(space.clone()),
                    comptime!(mi / bm),
                    comptime!(ni / bn),
                );
                let sub = self.at(&region);
                match comptime!(sub.space.partitioner()) {
                    Partitioner::Final(_) => sub,
                    Partitioner::Level(_) => {
                        sub.fragment_window(comptime!(mi % bm), comptime!(ni % bn))
                    }
                }
            }
        }
    }
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
            Coverage::PlaneLanes => {
                panic!(
                    "Coverage::PlaneLanes: unresolved Unit lane count; launch through space.launcher(client)"
                )
            }
        },
    }
}

/// Classify the current level of a space that backs plane tiles: `None` for an *instance* level
/// (a `Spatial` split across hardware, one tile per instance), or the trailing-two-axes tile
/// counts for the *partition* level — a purely sequential level is one even at a 1×1 grid (cuts
/// equal to the level below still back per-instance tiles). Anything else panics at comptime.
pub(crate) fn partition_level(space: &Space) -> Option<(usize, usize)> {
    if space.is_final() {
        return None;
    }
    if space.axes().any(|a| {
        matches!(
            space.partitioner().distribution(a),
            Distribution::Spatial { .. }
        )
    }) {
        for axis in space.axes() {
            assert!(
                per_instance_tiles(space, axis) == Some(1),
                "plane instance level: every axis must hand out one tile"
            );
        }
        return None;
    }
    let rank = space.rank();
    for (p, axis) in space.axes().enumerate() {
        let tiles = per_instance_tiles(space, axis)
            .expect("plane partition level: tile counts must be comptime");
        assert!(
            p >= rank - 2 || tiles == 1,
            "plane partition level: leading (batch) axes must hand out one tile"
        );
    }
    Some((
        per_instance_tiles(space, space.axis_at(rank - 2)).unwrap(),
        per_instance_tiles(space, space.axis_at(rank - 1)).unwrap(),
    ))
}

/// The whole remaining walk's tile grid for one instance: `(1, 1)` when every level is an instance
/// level, else the componentwise product of the partition levels' tile counts (the grid may be
/// split across stacked levels, e.g. an N-walk staging level over an M-only static walk).
pub(crate) fn partition_shape(space: &Space) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut level = space.clone();
    while !level.is_final() {
        if let Some((m, n)) = partition_level(&level) {
            shape = (shape.0 * m, shape.1 * n);
        }
        level = level.divide();
    }
    shape
}
