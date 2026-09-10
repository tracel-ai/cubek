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
    /// instruction contracts through. `vector_size` and `fold` shape the software block's lines
    /// ([`RegisterData::fold`]); the hardware encodings have no say in their layout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acc(
        #[comptime] form: PlaneForm,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            PlaneForm::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(
                MatrixIdent::Accumulator,
                m,
                n,
                k,
                MatrixLayout::RowMajor,
            )),
            PlaneForm::Mma { io } => {
                PlaneTile::new_Mma(MmaData::<T>::acc(m, n, k, MatrixLayout::RowMajor, io))
            }
            PlaneForm::Registers { config } => PlaneTile::new_Register(RegisterData::<T>::alloc(
                m,
                n,
                axes,
                vector_size,
                fold,
                config,
                monoid,
            )),
        }
    }

    /// An operand tile in role `ident`, uninitialized, loaded in `layout`: the role's own row
    /// order, or its transpose where the operand lies that way. `k` is the operand's own
    /// contraction depth, not the instruction's.
    pub(crate) fn operand(
        #[comptime] form: PlaneForm,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            PlaneForm::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k, layout)),
            PlaneForm::Mma { io } => match comptime!(ident) {
                MatrixIdent::A => PlaneTile::new_Mma(MmaData::<T>::lhs(m, n, k, layout, io)),
                MatrixIdent::B => PlaneTile::new_Mma(MmaData::<T>::rhs(m, n, k, layout, io)),
                MatrixIdent::Accumulator => {
                    panic!("PlaneTile::operand: an accumulator is not an operand")
                }
            },
            PlaneForm::Registers { .. } => {
                panic!("PlaneTile::operand: the software form stages no operand plane tile")
            }
        }
    }

    /// This tile carrying the scratch its partition was opened with, so a fragment taken off the
    /// partition can bounce on its own. Only a cmma tile bounces.
    pub(crate) fn with_scratch(
        self,
        scratch: Shared<[T]>,
        #[comptime] lanes: usize,
    ) -> PlaneTile<T> {
        match self {
            PlaneTile::Cmma(d) => PlaneTile::new_Cmma(d.with_scratch(scratch, lanes)),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::with_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Spill this tile into its slot of the plane's scratch. Only a cmma tile bounces.
    pub(crate) fn spill_to_scratch(&self) {
        match self {
            PlaneTile::Cmma(d) => d.spill_to_scratch(),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::spill_to_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Add this tile's spilled cells into `mem`. The other half of [`spill_to_scratch`].
    ///
    /// [`spill_to_scratch`]: Self::spill_to_scratch
    pub(crate) fn add_from_scratch<Out: Numeric>(
        &self,
        mem: &mut MemData<Out>,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => d.add_from_scratch(mem, space),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::add_from_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// The tile's `(m, n)`.
    pub(crate) fn shape(&self) -> comptime_type!((usize, usize)) {
        match self {
            PlaneTile::Cmma(d) => comptime!(d.shape),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::shape: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Store this tile, row-major, into `scratch` (one tile's cells).
    pub(crate) fn store_scratch(&self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.store_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::store_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch(&mut self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.load_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => {
                panic!("PlaneTile::load_scratch: only a cmma tile bounces through a scratch")
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

    pub(crate) fn scale(&mut self, factor: T) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                "PlaneTile::scale: a hardware mma fragment is not read cell by cell, so a scale \
                 over one folds at the store instead"
            ),
            PlaneTile::Register(d) => d.scale(factor),
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

    /// `space` is the sink window's: a hardware fragment is exactly the instruction's shape and
    /// stores through its own intrinsic, so only the software block and a cmma fragment draining
    /// into a store that folds read it.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut MemData<Out>,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => match comptime!(mem.access.write) {
                Write::Replace => d.store_cast_window(mem),
                Write::Accumulate => d.accumulate_cast_window(mem, space),
            },
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
    /// This plane's window of shared memory, one tile wide, that a row-wise op bounces a tile
    /// through: a tile's cells are not addressable in registers. Opened by
    /// [`with_scratch`](Tile::with_scratch); a partition without one contracts and drains only.
    pub scratch: ComptimeOption<Shared<[T]>>,
    /// How much of this partition the scratch holds, which is what tells a drain whether it may
    /// hoist its barriers out of the per-tile loop ([`Resident::drains_together`]).
    #[cube(comptime)]
    pub resident: Resident,
}

#[cube]
impl<T: Numeric> PlanePartition<T> {
    /// The `(mi, ni)` tile (a handle clone). Comptime indices only: plane tiles cannot be
    /// selected at runtime.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> PlaneTile<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The one fragment of a `1 × 1` partition. A grid of them has no single fragment: a store
    /// walks its cells, one fragment per region.
    pub(crate) fn fragment(&self) -> PlaneTile<T> {
        comptime!(assert!(
            self.m_tiles == 1 && self.n_tiles == 1,
            "PlanePartition::fragment: {} x {} fragments are stored one at a time; walk the \
             cells and copy each `acc.at(&cell)`",
            self.m_tiles,
            self.n_tiles
        ));
        self.at(0usize, 0usize)
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
            scratch: self.scratch.clone(),
            resident: comptime!(self.resident),
        }
    }

    /// `self[r, :] *= corr[r]` over the partition's rows, each tile bounced through the scratch:
    /// stored, scaled a cell per lane, loaded back.
    ///
    /// The syncs are cube-wide, as every other fragment bounce here is: a plane sync does not
    /// order a fragment store against the lanes' own writes on every backend. They sit outside
    /// the skip, so a plane whose factors are all one still reaches each one, and only its work
    /// is skipped; `corr` is plane-uniform, so that skip is.
    pub(crate) fn rescale_rows(&self, corr: &Array<T>, #[comptime] lanes: usize) {
        let mut scratch = #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.clone(),
            ComptimeOption::None => panic!(
                "PlanePartition::rescale_rows: a row-wise op on a plane-resident accumulator \
                 bounces through a scratch; open the accumulator with `with_scratch`"
            ),
        };
        let (m, n) = self.at(0usize, 0usize).shape();
        let rows = comptime!(self.m_tiles * m);
        let mut moved = false;
        #[unroll]
        for r in 0..rows {
            if corr[r] != T::from_int(1) {
                moved = true;
            }
        }
        let cells = comptime!(m * n);
        let lane = UNIT_POS_X as usize % lanes;
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut tile = self.at(mi, ni);
                if moved {
                    tile.store_scratch(&scratch);
                }
                sync_cube();
                if moved {
                    #[unroll]
                    for t in 0..comptime!(cells.div_ceil(lanes)) {
                        let cell = lane + t * lanes;
                        if comptime!(cells.is_multiple_of(lanes)) || cell < cells {
                            scratch[cell] *= corr[mi * m + cell / n];
                        }
                    }
                }
                sync_cube();
                if moved {
                    tile.load_scratch(&scratch);
                }
                sync_cube();
            }
        }
    }

    /// The plane-resident form of an accumulator over `space`: a partition mirroring its grid,
    /// tiles uninitialized. Opening the scope is purely structural; the caller states the init.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mirror(
        #[comptime] space: Space,
        #[comptime] axes: MatrixAxes,
        #[comptime] form: PlaneForm,
        #[comptime] fragments: Fragments,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
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
                    fold,
                    monoid,
                ));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles,
                n_tiles,
                scratch: ComptimeOption::new_None(),
                resident: Resident::None,
            }),
            // The space of the tile it mirrors: what the levels below it cut, as they cut it.
            // The fragments were sized from the statement alone, so a `Dynamic` extent here is
            // never read; the first partition level below reads its own edges off its child.
            space,
            depth,
            levels,
        }
    }

    /// The store for one region of an operand under `out`'s contraction: a partition mirroring
    /// the accumulator's `grid` of fragments along the operand's own axes, tiles uninitialized
    /// in `form`; [`copy_from`](Tile::copy_from) fills it. `m`/`n` are the accumulator fragment's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store(
        #[comptime] window: Space,
        #[comptime] form: PlaneForm,
        #[comptime] out: Space,
        #[comptime] grid: (usize, usize),
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile<T> {
        let a0 = comptime!(window.axis_at(window.rank() - 2));
        let a1 = comptime!(window.axis_at(window.rank() - 1));

        // The operand's role is which of the accumulator's axes it shares: `A` spans the rows,
        // `B` the columns. Its fragments run along that axis, one deep along the contraction —
        // counted in the window's own axis order, since that is how `at` addresses them.
        let contracted = comptime!(window.contraction(&out));
        let free = comptime!(if contracted == a1 {
            a0
        } else {
            assert!(
                contracted == a0,
                "PlanePartition::store: the contracted axis must be one of the trailing two"
            );
            a1
        });
        let out_rows = comptime!(out.axis_at(out.rank() - 2));
        let (ident, tiles) = comptime!(if free == out_rows {
            (MatrixIdent::A, grid.0)
        } else {
            assert!(
                free == out.axis_at(out.rank() - 1),
                "PlanePartition::store: the operand's free axis must be one of the output's \
                 trailing two"
            );
            (MatrixIdent::B, grid.1)
        });
        let (t0, t1) = comptime!(if free == a0 { (tiles, 1) } else { (1, tiles) });
        // The role's rows: `A` is `m×k` and `B` is `k×n`, so an operand whose window lists the
        // axes in that order is row-major, and one listing them the other way — a weight
        // stored `{n, k}`, read in lines along its contraction — is the same fragment loaded
        // col-major off the same rows.
        let layout = comptime!(if (ident == MatrixIdent::A) == (contracted == a1) {
            MatrixLayout::RowMajor
        } else {
            MatrixLayout::ColMajor
        });
        let k = comptime!(window.extent(contracted));

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _i in 0..t0 {
            #[unroll]
            for _j in 0..t1 {
                frags.push(PlaneTile::<T>::operand(form, ident, m, n, k, layout));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles: t0,
                n_tiles: t1,
                scratch: ComptimeOption::new_None(),
                resident: Resident::None,
            }),
            space: comptime!(window),
            depth,
            levels,
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
            comptime!(src.depth),
            comptime!(src.levels.clone()),
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
                let window = src.at(&Region::trailing(
                    comptime!(src.depth),
                    comptime!(src.space.clone()),
                    comptime!(level.clone()),
                    mi,
                    ni,
                ));
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

    /// Multiply every tile by `factor`.
    pub(crate) fn scale(&self, factor: T) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                frag.scale(factor);
            }
        }
    }
}

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
    Level::walk(&cuts)
}
