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

/// One plane-level tile, by encoding ([`PlaneForm`]).
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acc(
        #[comptime] form: PlaneForm,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] monoid: Monoid,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            PlaneForm::Cmma => {
                PlaneTile::new_Cmma(CmmaData::<T>::alloc(MatrixIdent::Accumulator, m, n, k))
            }
            PlaneForm::Mma { io } => {
                PlaneTile::new_Mma(MmaData::<T>::acc(m, n, k, MatrixLayout::RowMajor, io))
            }
            // `vector_size` is the promoting tile's, so the block's lines match the memory it
            // will drain into; the hardware encodings above have no say in their layout.
            PlaneForm::Registers { config } => PlaneTile::new_Register(RegisterData::<T>::alloc(
                m,
                n,
                axes,
                vector_size,
                config,
                monoid,
            )),
        }
    }

    /// An operand tile in role `ident`, uninitialized. `k` is the operand's own contraction
    /// depth, not the instruction's.
    pub(crate) fn operand(
        #[comptime] form: PlaneForm,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            PlaneForm::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k)),
            PlaneForm::Mma { io } => match comptime!(ident) {
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
            PlaneForm::Registers { .. } => {
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
            PlaneTile::Cmma(d) => {
                comptime!(mem.access.write.validate_fragment_drain("PlaneTile::Cmma"));
                d.store_cast_window(mem)
            }
            PlaneTile::Mma(d) => {
                comptime!(mem.access.write.validate_fragment_drain("PlaneTile::Mma"));
                d.store_cast_window(mem)
            }
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
        #[comptime] axes: MatrixAxes,
        #[comptime] form: PlaneForm,
        #[comptime] fragments: Fragments,
        #[comptime] vector_size: usize,
        #[comptime] monoid: Monoid,
    ) -> Tile<T> {
        let (m_tiles, n_tiles) = comptime!((fragments.m_tiles, fragments.n_tiles));
        let (m, n, k) = comptime!((fragments.m, fragments.n, fragments.k));

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                frags.push(PlaneTile::<T>::acc(
                    form,
                    m,
                    n,
                    axes,
                    k,
                    vector_size,
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
            // The space of the tile it mirrors: what the levels below it cut, as they cut it.
            // The fragments were sized from the statement alone, so a `Dynamic` extent here is
            // never read; the first partition level below reads its own edges off its child.
            space,
            descent: comptime!(Descent::root()),
        }
    }

    /// The store for one region of an operand under `out`'s contraction: a partition mirroring
    /// the accumulator's `grid` of fragments along the operand's own axes, tiles uninitialized
    /// in `form`; [`copy_from`](Tile::copy_from) fills it. `m`/`n` are the accumulator fragment's.
    pub(crate) fn store(
        #[comptime] window: Space,
        #[comptime] form: PlaneForm,
        #[comptime] out: Space,
        #[comptime] grid: (usize, usize),
        #[comptime] m: usize,
        #[comptime] n: usize,
    ) -> Tile<T> {
        let a0 = comptime!(window.axis_at(window.rank() - 2));
        let a1 = comptime!(window.axis_at(window.rank() - 1));

        // `A` is `m×k`, `B` is `k×n`: the operand's role is where its contracted axis sits, and
        // its fragments run along the accumulator's rows or columns accordingly, one deep along
        // the contraction.
        let contracted = comptime!(window.contraction(&out));
        let (ident, t0, t1) = comptime!(if contracted == a1 {
            (MatrixIdent::A, grid.0, 1)
        } else {
            assert!(
                contracted == a0,
                "PlanePartition::store: the contracted axis must be one of the trailing two"
            );
            (MatrixIdent::B, 1, grid.1)
        });
        let k = comptime!(window.extent(contracted));

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
            descent: comptime!(Descent::root()),
        }
    }

    /// This region of an operand loaded into cmma fragments, one per final tile of its grid,
    /// built where the kernel reads it. `acc` is the accumulator the fragments contract into,
    /// which fixes the fragment shape and the operand's role.
    pub fn cmma_fragments<Acc: Numeric>(src: &Tile<T>, acc: &Tile<Acc>) -> Tile<T> {
        PlanePartition::<T>::fragments_in(src, acc, comptime!(PlaneForm::Cmma))
    }

    /// [`cmma_fragments`](PlanePartition::cmma_fragments) in the manual-mma encoding, loaded by
    /// `io`'s transports.
    pub fn mma_fragments<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] io: MmaIOConfig,
    ) -> Tile<T> {
        PlanePartition::<T>::fragments_in(src, acc, comptime!(PlaneForm::Mma { io }))
    }

    fn fragments_in<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] form: PlaneForm,
    ) -> Tile<T> {
        let gathered = src.gathered();
        comptime!(assert!(
            !gathered,
            "PlanePartition::fragments: a gathered operand cannot load into fragments; stage it \
             into shared memory first"
        ));
        let (grid, m, n) = acc.fragment_grid();
        let mut frags = PlanePartition::<T>::store(
            comptime!(src.space.clone()),
            comptime!(form),
            comptime!(acc.space.clone()),
            comptime!(grid),
            comptime!(m),
            comptime!(n),
        );
        frags.copy_from(src);
        frags
    }

    /// Fill each tile from its window of `src`, in the partition's row-major order. The cut
    /// that turns the source's window into fragments is this partition's grid, one level the
    /// kernel never walks.
    pub(crate) fn fill_from(&self, src: &Tile<T>) {
        let level = comptime!(fragment_level(&src.space, self.m_tiles, self.n_tiles));
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.fragment_window(mi, ni, comptime!(vec![level.clone()]));
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
    /// `levels` is what this partition was descended with ([`Descent`]): the drain finds each
    /// window by replaying the nest the kernel walked the accumulator through.
    pub(crate) fn drain_into(&self, dst: &mut Tile<T>, #[comptime] levels: Vec<Level>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni, comptime!(levels.clone()));
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
    /// a plane accumulator (e.g. `f32`) written to a narrower output (e.g. `f16`). `levels` is
    /// what this partition was descended with ([`Descent`]).
    pub(crate) fn drain_cast_into<Out: Numeric>(
        &self,
        dst: &mut Tile<Out>,
        #[comptime] levels: Vec<Level>,
    ) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni, comptime!(levels.clone()));
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
    /// Descend to the `(mi, ni)` tile's final window through `levels`, the nest the accumulator
    /// was walked with: an instance level hands this instance a single region; a partition level
    /// takes its own digit of the grid coordinates: the grid may be split across stacked levels,
    /// so each consumes the high digits (the levels below it are the place value) and passes the
    /// rest down.
    pub(crate) fn fragment_window(
        &self,
        #[comptime] mi: usize,
        #[comptime] ni: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile<T> {
        let space = comptime!(self.space.clone());
        comptime!(assert!(
            !levels.is_empty(),
            "Tile::fragment_window: a final tile has no partition level to descend"
        ));
        let level = comptime!(levels[0].clone());
        let rest = comptime!(levels[1..].to_vec());
        match comptime!(level.role()) {
            // An instance level hands this instance one region; descend into it.
            LevelRole::Instance => {
                let walk = self.runtime_space().level(comptime!(level.clone()));
                let sub = self.at(&walk.region(0));
                if comptime!(rest.is_empty()) {
                    sub
                } else {
                    sub.fragment_window(mi, ni, rest)
                }
            }
            // A partition level takes its own digit of the grid and passes the rest down
            // (the grid may be split across stacked levels).
            LevelRole::Partition => {
                let (bm, bn) = comptime!(partition_shape(&level.child(&space), &rest));
                let region = Region::trailing(
                    comptime!(space.clone()),
                    comptime!(level.clone()),
                    comptime!(mi / bm),
                    comptime!(ni / bn),
                );
                let sub = self.at(&region);
                if comptime!(rest.is_empty()) {
                    sub
                } else {
                    sub.fragment_window(comptime!(mi % bm), comptime!(ni % bn), rest)
                }
            }
        }
    }
}

/// The tile grid `levels` cut one instance's `space` into: `(1, 1)` when every level is an
/// instance level, else the componentwise product of the partition levels' tile counts (the grid
/// may be split across stacked levels, e.g. an N-walk staging level over an M-only static walk).
pub(crate) fn partition_shape(space: &Space, levels: &[Level]) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut space = space.clone();
    for level in levels {
        // Only a partition level contributes a grid; an instance level spreads across hardware.
        match level.role() {
            LevelRole::Partition => {
                let (m, n) = level.partition_grid(&space);
                shape = (shape.0 * m, shape.1 * n);
            }
            LevelRole::Instance => {}
        }
        space = level.child(&space);
    }
    shape
}

/// The one level that cuts an operand's window into a `m_tiles × n_tiles` grid of fragments on
/// its trailing two axes, every other axis whole: what a partition fills from, and the kernel
/// never walks.
fn fragment_level(window: &Space, m_tiles: usize, n_tiles: usize) -> Level {
    let rank = window.rank();
    let axes: Vec<Axis> = window.axes().collect();
    let cuts: Vec<(Axis, usize)> = axes
        .iter()
        .enumerate()
        .map(|(p, &axis)| {
            let extent = window.extent(axis);
            let tiles = match p {
                p if p == rank - 2 => m_tiles,
                p if p == rank - 1 => n_tiles,
                _ => 1,
            };
            assert!(
                extent.is_multiple_of(tiles),
                "PlanePartition::fill_from: {tiles} fragments do not divide the {extent} of {axis:?}"
            );
            (axis, extent / tiles)
        })
        .collect();
    Level::cuts(&axes, |l| {
        l.walk(&cuts);
    })
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// The fragment grid this accumulator holds and one fragment's `m × n`: a partition's own
    /// grid, a single plane tile's `1 × 1` of its whole window.
    pub(crate) fn fragment_grid(&self) -> comptime_type!(((usize, usize), usize, usize)) {
        let rank = comptime!(self.space.rank());
        let rows = comptime!(self.space.extent_at(rank - 2));
        let cols = comptime!(self.space.extent_at(rank - 1));
        match &self.tile_kind {
            TileKind::PlanePartition(p) => {
                comptime!(((p.m_tiles, p.n_tiles), rows / p.m_tiles, cols / p.n_tiles))
            }
            TileKind::PlaneTile(_) => comptime!(((1, 1), rows, cols)),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::fragment_grid: only a plane-resident accumulator holds fragments")
            }
        }
    }
}
