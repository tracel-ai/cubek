//! The plane-level tile ([`PlaneTile`]) and the grid of them one plane owns
//! ([`PlanePartition`]).
//!
//! A plane tile is owned by a plane and sliced across its lanes, never unit-addressable: cmma's
//! `Matrix` is `MatrixScope::Plane`, manual mma's registers index by `UNIT_POS_PLANE`. One concept,
//! two encodings ([`CmmaData`], [`MmaData`]), so the partition over them is written once.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// One plane-level tile, by encoding ([`Instruction`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum PlaneTile<T: Numeric> {
    Cmma(CmmaData<T>),
    Mma(MmaData<T>),
    /// The software leaf's accumulator: a register block, not a hardware fragment. Its lanes
    /// hold the block the way the encoding above holds a matrix, so the partition over the
    /// three stays encoding-blind.
    Registers(RegisterData<T>),
}

#[cube]
impl<T: Numeric> PlaneTile<T> {
    /// An accumulator tile over the whole `m × n` MMA tile, uninitialized, in the `form` the
    /// instruction contracts through. `vector_size` and `fold` shape the software block's lines
    /// ([`RegisterData::fold`]); the hardware encodings have no say in their layout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acc(
        #[comptime] form: Instruction,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(
                MatrixIdent::Accumulator,
                m,
                n,
                k,
                MatrixLayout::RowMajor,
            )),
            Instruction::Mma { io } => {
                PlaneTile::new_Mma(MmaData::<T>::acc(m, n, k, MatrixLayout::RowMajor, io))
            }
            Instruction::Registers { config } => PlaneTile::new_Registers(
                RegisterData::<T>::alloc(m, n, axes, vector_size, fold, config, monoid),
            ),
        }
    }

    /// An operand tile in role `ident`, uninitialized, loaded in `layout`: the role's own row
    /// order, or its transpose where the operand lies that way. `k` is the operand's own
    /// contraction depth, not the instruction's.
    pub(crate) fn operand(
        #[comptime] form: Instruction,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k, layout)),
            Instruction::Mma { io } => match comptime!(ident) {
                MatrixIdent::A => PlaneTile::new_Mma(MmaData::<T>::lhs(m, n, k, layout, io)),
                MatrixIdent::B => PlaneTile::new_Mma(MmaData::<T>::rhs(m, n, k, layout, io)),
                MatrixIdent::Accumulator => {
                    panic!("PlaneTile::operand: an accumulator is not an operand")
                }
            },
            Instruction::Registers { .. } => {
                panic!("PlaneTile::operand: the software form stages no operand plane tile")
            }
        }
    }

    /// This tile carrying the scratch its partition was opened with, so a fragment taken off the
    /// partition can bounce on its own. Only a cmma tile bounces.
    pub(crate) fn with_scratch(self, scratch: Shared<[T]>) -> PlaneTile<T> {
        match self {
            PlaneTile::Cmma(d) => PlaneTile::new_Cmma(d.with_scratch(scratch)),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::with_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Spill this tile into its slot of the plane's scratch. Only a cmma tile bounces.
    pub(crate) fn spill_to_scratch(&self) {
        match self {
            PlaneTile::Cmma(d) => d.spill_to_scratch(),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::spill_to_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Add this tile's spilled cells into `mem`. The other half of [`spill_to_scratch`].
    ///
    /// [`spill_to_scratch`]: Self::spill_to_scratch
    pub(crate) fn add_from_scratch<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => d.add_from_scratch(mem, space),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::add_from_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// The tile's `(m, n)`, which only a cmma tile carries: the two other encodings size
    /// themselves off the instruction and never bounce through a scratch.
    pub(crate) fn shape(&self) -> comptime_type!((usize, usize)) {
        match self {
            PlaneTile::Cmma(d) => comptime!(d.shape),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::shape: only a cmma tile states its shape")
            }
        }
    }

    /// Store this tile, row-major, into `scratch` (one tile's cells).
    pub(crate) fn store_scratch(&self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.store_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::store_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch(&mut self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.load_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlaneTile::load_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    pub(crate) fn zero(&mut self) {
        match self {
            PlaneTile::Cmma(d) => d.zero(),
            PlaneTile::Mma(d) => d.zero(),
            PlaneTile::Registers(d) => d.zero(),
        }
    }

    pub(crate) fn init(&mut self, val: T) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => {
                panic!("PlaneTile::init: a hardware mma fragment has no fill other than zero")
            }
            PlaneTile::Registers(d) => d.init(val),
        }
    }

    pub(crate) fn scale(&mut self, factor: T) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                "PlaneTile::scale: a hardware mma fragment is not read cell by cell, so a scale \
                 over one folds at the store instead"
            ),
            PlaneTile::Registers(d) => d.scale(factor),
        }
    }

    /// Fill this fragment from a memory `src`. Takes the whole tile, not its store: the manual-mma
    /// transport reads element by element through the quant-transparent matrix view, so it needs
    /// the space that view is shaped by. A cmma load takes the raw window and cannot decode.
    pub(crate) fn load_window(&mut self, src: &Tile<T>) {
        match self {
            PlaneTile::Cmma(d) => match &src.kind {
                TileKind::Memory(m) => {
                    d.load_window(m, comptime!(MatrixAxes::edges(&src.place.space).row_split))
                }
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("PlaneTile::load_window: a cmma fragment loads from memory")
                }
            },
            PlaneTile::Mma(d) => d.load_window(src),
            // Only an accumulator takes this encoding, and an accumulator is filled by
            // `zero` or by contracting into it, never by loading an operand window.
            PlaneTile::Registers(_) => {
                panic!("PlaneTile::load_window: a register accumulator is not a fill sink")
            }
        }
    }

    pub(crate) fn store_window(&self, mem: &mut Memory<T>, #[comptime] space: Space) {
        match self {
            PlaneTile::Cmma(d) => {
                d.store_window(mem, comptime!(MatrixAxes::edges(&space).row_split))
            }
            PlaneTile::Mma(d) => d.store_window(mem),
            // Same-type store; the block drains through `store_cast_window`, which is the
            // same write with the cast the wider accumulator needs.
            PlaneTile::Registers(d) => d.store_cast_window(mem, space),
        }
    }

    /// Add `src`'s block into this one, casting up. Only the register encoding promotes: a
    /// hardware fragment's element is the instruction's, so there is no narrow twin of it to
    /// drain, and the pairing is a plan's mistake rather than a shape this could serve.
    pub(crate) fn add_cast_from<S: Numeric>(&mut self, src: &PlaneTile<S>) {
        match (self, src) {
            (PlaneTile::Registers(d), PlaneTile::Registers(s)) => d.add_cast_from(s),
            _ => panic!(
                "PlaneTile::add_cast_from: a register block promotes into a register block; a \
                 fragment accumulates in the element its instruction names"
            ),
        }
    }

    /// `space` is the sink window's: a hardware fragment is exactly the instruction's shape and
    /// stores through its own intrinsic, so only the software block and a cmma fragment draining
    /// into a store that folds read it.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => match comptime!(mem.access.write) {
                Write::Replace => {
                    d.store_cast_window(mem, comptime!(MatrixAxes::edges(&space).row_split))
                }
                Write::Accumulate => d.accumulate_cast_window(mem, space),
            },
            PlaneTile::Mma(d) => {
                comptime!(mem.access.write.validate_fragment_drain("PlaneTile::Mma"));
                d.store_cast_window(mem)
            }
            PlaneTile::Registers(d) => d.store_cast_window(mem, space),
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
    /// One tile's edges along the window's trailing two axes: the leaf the grid is
    /// `m_tiles × n_tiles` of, which is what a fill cuts the window by.
    #[cube(comptime)]
    pub rows: usize,
    #[cube(comptime)]
    pub cols: usize,
    /// This plane's window of shared memory, one tile wide, that a row-wise op bounces a tile
    /// through: a tile's cells are not addressable in registers. Opened by
    /// [`with_scratch`](Tile::with_scratch); a partition without one contracts and drains only.
    pub scratch: ComptimeOption<Shared<[T]>>,
    /// How much of this grid the scratch holds, which is what tells a drain whether it may hoist
    /// its barriers out of the per-tile loop ([`DrainPlan`](crate::DrainPlan)).
    #[cube(comptime)]
    pub held: Scratch,
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

    /// One level down: the `sub_m × sub_n` block of fragments `step` selects, or the one
    /// fragment where the block is `1 × 1`.
    ///
    /// A partition selects under comptime coordinates (an unrolled walk folds regions to
    /// constants); an uncut level selects the whole partition. A runtime coordinate reaches here
    /// only from a `Dynamic` (top) level, which cuts nothing on `m`/`n`; a rolled *cut* is refused.
    pub(crate) fn at_step(&self, step: &Step, #[comptime] space: Space) -> TileKind<T> {
        let edges = comptime!(MatrixAxes::edges(&space));
        let a0 = comptime!(space.axis_at(edges.row_split));
        let a1 = comptime!(space.axis_at(edges.col_split));
        // A single-tile static axis (k-step, no m/n cut) folds to constant `0`, so a cut axis
        // takes its constant digit and an uncut one selects the whole partition. A `Dynamic`
        // axis (top level only) stays runtime, yielding `None`.
        let mi = if comptime!(step.level.tiles_const(&space, a0) == Some(1)) {
            comptime!(Some(0u64))
        } else {
            step.coord(a0).constant()
        };
        let ni = if comptime!(step.level.tiles_const(&space, a1) == Some(1)) {
            comptime!(Some(0u64))
        } else {
            step.coord(a1).constant()
        };
        match comptime!(mi.zip(ni)) {
            Some((c0, c1)) => {
                let (sub_m, sub_n) = comptime!({
                    let (cm, cn) = (step.level.tiles(&space, a0), step.level.tiles(&space, a1));
                    assert!(
                        self.m_tiles.is_multiple_of(cm) && self.n_tiles.is_multiple_of(cn),
                        "Tile::at: the level's grid must divide the partition"
                    );
                    (self.m_tiles / cm, self.n_tiles / cn)
                });
                let mi = comptime!(c0 as usize * sub_m);
                let ni = comptime!(c1 as usize * sub_n);
                if comptime!(sub_m == 1 && sub_n == 1) {
                    TileKind::new_PlaneTile(self.at(mi, ni))
                } else {
                    TileKind::new_PlanePartition(self.window(mi, ni, sub_m, sub_n))
                }
            }
            None => {
                comptime!(assert!(
                    !MatrixGrid::new(&step.level, &space).cuts(),
                    "Tile::at: a level that cuts a partition must be walked with compile-time \
                     coordinates (an unrolled walk)"
                ));
                TileKind::new_PlanePartition(self.clone())
            }
        }
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
            rows: comptime!(self.rows),
            cols: comptime!(self.cols),
            scratch: self.scratch.clone(),
            held: comptime!(self.held),
        }
    }

    /// `self[r, :] *= corr[r]` over the partition's rows, each tile bounced through the scratch:
    /// stored, scaled a cell per lane, loaded back.
    ///
    /// The syncs are cube-wide, as every fragment bounce here is: a plane sync does not order a
    /// fragment store against the lanes' own writes on every backend. They sit outside the skip,
    /// so a plane whose factors are all one still reaches each; the skip is uniform, as `corr` is.
    pub(crate) fn rescale_rows(&self, corr: &Array<T>) {
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
        // The plane's own width and this unit's place in it, both the hardware's: the lanes
        // deal the cells between them whatever shape the launch gave the cube.
        let lanes = PLANE_DIM as usize;
        let lane = UNIT_POS_PLANE as usize;
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
                    let mut cell = lane;
                    while cell < cells {
                        scratch[cell] *= corr[mi * m + cell / n];
                        cell += lanes;
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
        #[comptime] form: Instruction,
        #[comptime] grid: GridShape,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] monoid: Monoid,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile<T> {
        let (m_tiles, n_tiles) = comptime!(grid.tiles);
        let (m, n, k) = comptime!((grid.m, grid.n, grid.k));

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
            kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles,
                n_tiles,
                rows: m,
                cols: n,
                scratch: ComptimeOption::new_None(),
                held: Scratch::None,
            }),
            // The space of the tile it mirrors: what the levels below it cut, as they cut it.
            // The fragments were sized from the statement alone, so a `Dynamic` extent here is
            // never read; the first partition level below reads its own edges off its child.
            place: comptime!(Placement::new(space, depth, levels)),
        }
    }

    /// The store for one region of an operand under `out`'s contraction: a partition mirroring
    /// the accumulator's `grid` of fragments along the operand's own axes, tiles uninitialized
    /// in `form`; [`copy_from`](Tile::copy_from) fills it. `m`/`n` are the accumulator fragment's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store(
        #[comptime] window: Space,
        #[comptime] form: Instruction,
        #[comptime] out: Space,
        #[comptime] grid: (usize, usize),
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile<T> {
        let edges = comptime!(MatrixAxes::edges(&window));
        let a0 = comptime!(window.axis_at(edges.row_split));
        let a1 = comptime!(window.axis_at(edges.col_split));

        // The operand's role is which accumulator axis it shares: `A` the rows, `B` the columns.
        // Its fragments run along that axis, one deep along the contraction, in the window's own
        // axis order (how `at` addresses them). The contracted axis is the one `out` lacks.
        let (contracted, free) = comptime!(match (out.contains(a0), out.contains(a1)) {
            (false, true) => (a0, a1),
            (true, false) => (a1, a0),
            _ => panic!(
                "PlanePartition::store: one of the window's matrix axes is contracted and the \
                 other is the accumulator's"
            ),
        });
        let out_edges = comptime!(MatrixAxes::edges(&out));
        let out_rows = comptime!(out.axis_at(out_edges.row_split));
        let (ident, tiles) = comptime!(if free == out_rows {
            (MatrixIdent::A, grid.0)
        } else {
            assert!(
                free == out.axis_at(out_edges.col_split),
                "PlanePartition::store: the operand's free axis must be one of the output's \
                 trailing two"
            );
            (MatrixIdent::B, grid.1)
        });
        let (t0, t1) = comptime!(if free == a0 { (tiles, 1) } else { (1, tiles) });
        // One fragment's edges along the window's trailing two axes: the role's rows along the
        // free axis, the whole contraction along the other.
        let rows_along_free = comptime!(if ident == MatrixIdent::A { m } else { n });
        let k = comptime!(window.extent(contracted));
        let (e0, e1) = comptime!(if free == a0 {
            (rows_along_free, k)
        } else {
            (k, rows_along_free)
        });
        // The role's rows: `A` is `m×k` and `B` is `k×n`, so a window listing the axes in that
        // order is row-major, and one listing them the other way (a weight stored `{n, k}`) is the
        // same fragment loaded col-major off the same rows.
        let layout = comptime!(if (ident == MatrixIdent::A) == (contracted == a1) {
            MatrixLayout::RowMajor
        } else {
            MatrixLayout::ColMajor
        });

        let mut frags = Sequence::<PlaneTile<T>>::new();
        #[unroll]
        for _i in 0..t0 {
            #[unroll]
            for _j in 0..t1 {
                frags.push(PlaneTile::<T>::operand(form, ident, m, n, k, layout));
            }
        }
        Tile::<T> {
            kind: TileKind::new_PlanePartition(PlanePartition::<T> {
                frags,
                m_tiles: t0,
                n_tiles: t1,
                rows: e0,
                cols: e1,
                scratch: ComptimeOption::new_None(),
                held: Scratch::None,
            }),
            place: comptime!(Placement::new(window, depth, levels)),
        }
    }

    /// This region of an operand, in the form `instruction` reads it.
    ///
    /// **The one loader a kernel whose instruction is data wants**, twin of
    /// [`Tile::accumulator`](crate::Tile::accumulator). A register block reads lines out of the
    /// tile it is handed, so the tile *is* the answer; the matrix forms want their own fragments.
    pub fn operand<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] instruction: Instruction,
    ) -> Tile<T> {
        match comptime!(instruction) {
            Instruction::Registers { .. } => src.clone(),
            Instruction::Cmma => PlanePartition::<T>::cmma_fragments(src, acc),
            Instruction::Mma { io } => PlanePartition::<T>::mma_fragments(src, acc, io),
        }
    }

    /// This region of an operand loaded into cmma fragments, one per final tile of its grid,
    /// built where the kernel reads it. `acc` is the accumulator the fragments contract into,
    /// which fixes the fragment shape and the operand's role.
    pub fn cmma_fragments<Acc: Numeric>(src: &Tile<T>, acc: &Tile<Acc>) -> Tile<T> {
        PlanePartition::<T>::fragments_in(src, acc, comptime!(Instruction::Cmma))
    }

    /// [`cmma_fragments`](PlanePartition::cmma_fragments) in the manual-mma encoding, loaded by
    /// `io`'s transports.
    pub fn mma_fragments<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] io: MmaIo,
    ) -> Tile<T> {
        PlanePartition::<T>::fragments_in(src, acc, comptime!(Instruction::Mma { io }))
    }

    fn fragments_in<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] form: Instruction,
    ) -> Tile<T> {
        let gathered = src.gathered();
        comptime!(assert!(
            !gathered,
            "PlanePartition::fragments: a gathered operand cannot load into fragments; stage it \
             into shared memory first"
        ));
        let (grid, m, n) = acc.fragment_grid();
        let mut frags = PlanePartition::<T>::store(
            comptime!(src.place.space.clone()),
            comptime!(form),
            comptime!(acc.place.space.clone()),
            comptime!(grid),
            comptime!(m),
            comptime!(n),
            comptime!(src.place.depth),
            comptime!(src.place.levels.clone()),
        );
        frags.copy_from(src);
        frags
    }

    /// Fill each tile from its window of `src`, in the partition's row-major order. The cut
    /// that turns the source's window into fragments is this partition's grid, one level the
    /// kernel never walks.
    pub(crate) fn fill_from(&self, src: &Tile<T>) {
        let level = comptime!(fragment_level(
            &src.place.space,
            (self.rows, self.cols),
            (self.m_tiles, self.n_tiles)
        ));
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.at(&Region::trailing(
                    comptime!(src.place.depth),
                    comptime!(src.place.space.clone()),
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
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.zero();
        }
    }

    /// Initialize every tile with `val`.
    pub(crate) fn init(&self, val: T) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.init(val);
        }
    }

    /// Multiply every tile by `factor`.
    pub(crate) fn scale(&self, factor: T) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.scale(factor);
        }
    }
}

/// The `rows × cols` grid of fragments the walked `levels` cut `space` into.
pub(crate) fn partition_shape(space: &Space, levels: &[Level]) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut space = space.clone();
    for level in levels {
        let grid = MatrixGrid::new(level, &space);
        shape = (shape.0 * grid.rows, shape.1 * grid.cols);
        space = level.child(&space);
    }
    shape
}

/// The `rows × cols` grid of fragments a walked level cuts a partition into, read off `space`'s
/// two matrix edges; leading (batch) axes must hand out one tile.
pub(crate) struct MatrixGrid {
    rows: usize,
    cols: usize,
}

impl MatrixGrid {
    /// A dealt level spreads its tiles across hardware and cuts the partition nothing: only a
    /// walked level's grid is a grid of fragments.
    pub(crate) fn new(level: &Level, space: &Space) -> Self {
        if level.takers() != Takers::Walk {
            return MatrixGrid { rows: 1, cols: 1 };
        }
        let edges = MatrixAxes::edges(space);
        for (p, axis) in space.axes().enumerate() {
            let tiles = level
                .tiles_const(space, axis)
                .expect("plane partition level: tile counts must be comptime");
            assert!(
                p == edges.row_split || p == edges.col_split || tiles == 1,
                "plane partition level: leading (batch) axes must hand out one tile"
            );
        }
        MatrixGrid {
            rows: level
                .tiles_const(space, space.axis_at(edges.row_split))
                .unwrap(),
            cols: level
                .tiles_const(space, space.axis_at(edges.col_split))
                .unwrap(),
        }
    }

    /// Whether the level cuts the partition into more than one fragment, so each region must be
    /// selected by a comptime coordinate. A dealt level and a degenerate 1×1 partition (a k-step
    /// walk) both cut nothing.
    pub(crate) fn cuts(&self) -> bool {
        (self.rows, self.cols) != (1, 1)
    }
}

/// The one level that cuts an operand's window into the partition's grid of fragments on its
/// trailing two axes, every other axis whole: what a partition fills from, never walked. Stated
/// from the leaf up (one fragment's edges, then how many) and held to the window it fills from.
pub(crate) fn fragment_level(window: &Space, frag: (usize, usize), tiles: (usize, usize)) -> Level {
    let edges = MatrixAxes::edges(window);
    let (p0, p1) = (edges.row_split, edges.col_split);
    let axes: Vec<Axis> = window.axes().collect();
    let leaf: Vec<(Axis, usize)> = axes
        .iter()
        .enumerate()
        .map(|(p, &axis)| match p {
            p if p == p0 => (axis, frag.0),
            p if p == p1 => (axis, frag.1),
            _ => (axis, window.extent(axis)),
        })
        .collect();
    let counts: Vec<(Axis, usize)> = axes
        .iter()
        .enumerate()
        .map(|(p, &axis)| match p {
            p if p == p0 => (axis, tiles.0),
            p if p == p1 => (axis, tiles.1),
            _ => (axis, 1),
        })
        .collect();
    for (&(axis, edge), &(_, count)) in leaf.iter().zip(&counts) {
        assert!(
            edge * count == window.extent(axis),
            "PlanePartition::fill_from: {count} fragments of {edge} do not cover the {} of {axis:?}",
            window.extent(axis)
        );
    }
    Levels::leaf(&leaf).walk(&counts).build().remove(0)
}
