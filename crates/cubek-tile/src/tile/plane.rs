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

/// One plane-level tile, by encoding. The [`Instruction`] picks which.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum PlaneTile<T: Numeric> {
    Cmma(CmmaData<T>),
    Mma(MmaData<T>),
    /// The software leaf's accumulator: a register block, not a hardware fragment. Its lanes
    /// hold the block the way the encoding above holds a matrix, so the partition over the
    /// three stays encoding-blind.
    Register(RegisterData<T>),
}

#[cube]
impl<T: Numeric> PlaneTile<T> {
    /// An accumulator tile over the whole `m × n` MMA tile, uninitialized, in the `form` the
    /// instruction contracts through.
    pub(crate) fn acc(
        #[comptime] form: Instruction,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] lane_share: LaneShare,
        #[comptime] monoid: Monoid,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            Instruction::Cmma => {
                PlaneTile::new_Cmma(CmmaData::<T>::alloc(MatrixIdent::Accumulator, m, n, k))
            }
            Instruction::Mma { io } => {
                PlaneTile::new_Mma(MmaData::<T>::acc(m, n, k, MatrixLayout::RowMajor, io))
            }
            // `vector_size` is the promoting tile's, so the block's lines match the memory it
            // will drain into; the hardware encodings above have no say in their layout.
            Instruction::Registers { config } => PlaneTile::new_Register(RegisterData::<T>::alloc(
                m,
                n,
                vector_size,
                lane_share,
                config,
                monoid,
            )),
        }
    }

    /// An operand tile in role `ident`, uninitialized. `k` is the operand's own contraction
    /// depth, not the instruction's.
    pub(crate) fn operand(
        #[comptime] form: Instruction,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k)),
            Instruction::Mma { io } => match comptime!(ident) {
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
            Instruction::Registers { .. } => {
                panic!("PlaneTile::operand: the software form stages no operand plane tile")
            }
        }
    }

    pub(crate) fn zero(&mut self) {
        match self {
            PlaneTile::Cmma(d) => d.zero(),
            PlaneTile::Mma(d) => d.zero(),
            PlaneTile::Register(d) => d.zero(),
        }
    }

    pub(crate) fn init(&mut self, val: T) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => {
                panic!("PlaneTile::init: a hardware mma fragment has no fill other than zero")
            }
            PlaneTile::Register(d) => d.init(val),
        }
    }

    /// Fill this fragment from a memory `src`. Takes the whole tile, not its store: the manual-mma
    /// transport reads element by element through the quant-transparent matrix view, so it needs
    /// the space that view is shaped by. A cmma load takes the raw window and cannot decode.
    pub(crate) fn load_window(&mut self, src: &Tile<T>) {
        match self {
            PlaneTile::Cmma(d) => match &src.tile_kind {
                TileKind::Gmem(m) | TileKind::Smem(m) => d.load_window(m),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_) => {
                    panic!("PlaneTile::load_window: a cmma fragment loads from memory")
                }
            },
            PlaneTile::Mma(d) => d.load_window(src),
            // Only an accumulator takes this encoding, and an accumulator is filled by
            // `zero` or by contracting into it, never by loading an operand window.
            PlaneTile::Register(_) => {
                panic!("PlaneTile::load_window: a register accumulator is not a fill sink")
            }
        }
    }

    pub(crate) fn store_window(&self, mem: &mut MemData<T>, #[comptime] space: Space) {
        match self {
            PlaneTile::Cmma(d) => d.store_window(mem),
            PlaneTile::Mma(d) => d.store_window(mem),
            // Same-type store; the block drains through `store_cast_window`, which is the
            // same write with the cast the wider accumulator needs.
            PlaneTile::Register(d) => d.store_cast_window(mem, space),
        }
    }

    /// `space` is the sink window's, and only the software block reads it: a hardware fragment
    /// is exactly the instruction's shape and stores through its own intrinsic.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut MemData<Out>,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => d.store_cast_window(mem),
            PlaneTile::Mma(d) => d.store_cast_window(mem),
            PlaneTile::Register(d) => d.store_cast_window(mem, space),
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
    /// tiles uninitialized. Opening the scope is purely structural; the caller states the init.
    pub(crate) fn mirror(
        #[comptime] space: Space,
        #[comptime] form: Instruction,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] lane_share: LaneShare,
        #[comptime] monoid: Monoid,
    ) -> Tile<T> {
        let (m_tiles, n_tiles) = comptime!(partition_shape(&space));
        let fin = comptime!(space.final_space());
        let m = comptime!(fin.extent_at(fin.rank() - 2));
        let n = comptime!(fin.extent_at(fin.rank() - 1));

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                frags.push(PlaneTile::<T>::acc(
                    form,
                    m,
                    n,
                    k,
                    vector_size,
                    lane_share,
                    monoid,
                ));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles,
                n_tiles,
            }),
            // The fragments above were sized from the partitioner alone (`partition_shape`
            // and `final_space` read edges, never extents), so the tile carries the space it
            // actually has, not the caller's. The kernel-form space is `all_dynamic`, and a
            // register-resident tile has no buffer bound to resolve a `Dynamic` axis from, and
            // inheriting it verbatim would make `runtime_space` unanswerable.
            space: comptime!(space.sub_tile_space()),
        }
    }

    /// The staging store for one region of an operand under `out`'s contraction: a partition
    /// mirroring the region's grid, tiles uninitialized in the `form` the operand's stages
    /// stated; [`copy_from`](Tile::copy_from) fills it.
    pub(crate) fn store(
        #[comptime] window: Space,
        #[comptime] form: Instruction,
        #[comptime] out: Space,
    ) -> Tile<T> {
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
                frags.push(PlaneTile::<T>::operand(form, ident, m, n, k));
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
                frag.load_window(&window);
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

    /// Initialize every tile with `val`.
    pub(crate) fn init(&self, val: T) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                frag.init(val);
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
                let space = comptime!(window.space.clone());
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g, space),
                    TileKind::PlaneTile(_)
                    | TileKind::PlanePartition(_)
                    | TileKind::TmaGmem(_)
                    | TileKind::Procedural(_) => {
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
                let space = comptime!(window.space.clone());
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_cast_window(g, space),
                    TileKind::PlaneTile(_)
                    | TileKind::PlanePartition(_)
                    | TileKind::TmaGmem(_)
                    | TileKind::Procedural(_) => {
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
    /// single region; a partition level takes its own digit of the grid coordinates: the grid may
    /// be split across stacked levels, so each consumes the high digits (the levels below it are
    /// the place value) and passes the rest down.
    pub(crate) fn fragment_window(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> Tile<T> {
        let space = comptime!(self.space.clone());
        match comptime!(space.partitioner().clone()) {
            // The recursion always terminates at a final store below, never by descending into one.
            Partitioner::Final => {
                panic!("Tile::fragment_window: a final tile has no partition level to descend")
            }
            Partitioner::Level(level) => match comptime!(level.role()) {
                // An instance level hands this instance one region; descend into it.
                LevelRole::Instance => {
                    let walk = Walk::over(self.runtime_space());
                    let sub = self.at(&walk.region(0));
                    match comptime!(sub.space.partitioner()) {
                        Partitioner::Final => sub,
                        Partitioner::Level(_) => sub.fragment_window(mi, ni),
                    }
                }
                // A partition level takes its own digit of the grid and passes the rest down
                // (the grid may be split across stacked levels).
                LevelRole::Partition => {
                    let (bm, bn) = comptime!(partition_shape(&space.divide()));
                    let region = Region::trailing(
                        comptime!(space.clone()),
                        comptime!(mi / bm),
                        comptime!(ni / bn),
                    );
                    let sub = self.at(&region);
                    match comptime!(sub.space.partitioner()) {
                        Partitioner::Final => sub,
                        Partitioner::Level(_) => {
                            sub.fragment_window(comptime!(mi % bm), comptime!(ni % bn))
                        }
                    }
                }
            },
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

/// The `m × n` grid a partition level cuts, read off its trailing two axes; leading (batch) axes
/// must hand out one tile. Valid only on a [`Partition`](LevelRole::Partition) level; the role
/// says whether it applies, this only reads the counts.
pub(crate) fn partition_grid(space: &Space) -> (usize, usize) {
    let rank = space.rank();
    for (p, axis) in space.axes().enumerate() {
        let tiles = per_instance_tiles(space, axis)
            .expect("plane partition level: tile counts must be comptime");
        assert!(
            p >= rank - 2 || tiles == 1,
            "plane partition level: leading (batch) axes must hand out one tile"
        );
    }
    (
        per_instance_tiles(space, space.axis_at(rank - 2)).unwrap(),
        per_instance_tiles(space, space.axis_at(rank - 1)).unwrap(),
    )
}

/// The whole remaining walk's tile grid for one instance: `(1, 1)` when every level is an instance
/// level, else the componentwise product of the partition levels' tile counts (the grid may be
/// split across stacked levels, e.g. an N-walk staging level over an M-only static walk).
pub(crate) fn partition_shape(space: &Space) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut level = space.clone();
    while !level.is_final() {
        // Only a partition level contributes a grid; an instance level spreads across hardware.
        match level.partitioner().role() {
            LevelRole::Partition => {
                let (m, n) = partition_grid(&level);
                shape = (shape.0 * m, shape.1 * n);
            }
            LevelRole::Instance => {}
        }
        level = level.divide();
    }
    shape
}
