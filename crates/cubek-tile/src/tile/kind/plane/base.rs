//! The plane-level tile ([`PlaneTile`]) and the grid of them one plane owns
//! ([`PlanePartition`]).

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::tile::slices::AxisSlicesKind;
use crate::*;

/// One plane-level tile, by encoding ([`Instruction`]).
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum PlaneTile<T: Numeric> {
    Cmma(CmmaData<T>),
    Mma(MmaData<T>),
    /// The software leaf's accumulator: a register block, not a hardware fragment.
    Registers(RegisterData<T>),
}

#[cube]
impl<T: Numeric> PlaneTile<T> {
    /// The semiring this tile contracts under: a hardware instruction runs the sum of products
    /// alone, a register block the one it was opened with.
    pub(crate) fn semiring(&self) -> comptime_type!(Semiring) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => comptime!(Semiring::SUM_PROD),
            PlaneTile::Registers(d) => comptime!(d.accumulation.semiring("PlaneTile::mma")),
        }
    }

    /// An uninitialized accumulator tile over the whole `m × n` MMA tile, in `form`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acc(
        #[comptime] form: Instruction,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] accumulation: Accumulation,
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
                RegisterData::<T>::alloc(m, n, axes, vector_size, fold, config, accumulation),
            ),
        }
    }

    /// An uninitialized operand tile in role `ident`, loaded in `layout`, block-scaled where
    /// `block_scaled` says so. `k` is the operand's own contraction depth, not the instruction's.
    pub(crate) fn operand(
        #[comptime] form: Instruction,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_scaled: bool,
    ) -> PlaneTile<T> {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::<T>::alloc(ident, m, n, k, layout)),
            Instruction::Mma { io } => PlaneTile::new_Mma(MmaData::<T>::operand(
                ident,
                m,
                n,
                k,
                layout,
                io,
                block_scaled,
            )),
            Instruction::Registers { .. } => {
                panic!("PlaneTile::operand: the software form stages no operand plane tile")
            }
        }
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the tile's columns: a register
    /// block in place, a cmma fragment bounced through its slot of the plane's scratch on
    /// `sync_plane`.
    pub(crate) fn mul_along(&self, factors: &Array<T>, #[comptime] first: usize) {
        match self {
            PlaneTile::Registers(block) => {
                let mut block = block.clone();
                block.mul_along(factors, first);
            }
            PlaneTile::Cmma(fragment) => fragment.mul_along(factors, first),
            PlaneTile::Mma(fragment) => {
                let mut fragment = fragment.clone();
                fragment.mul_rows(factors, first);
            }
        }
    }

    /// This accumulator tile as the left factor of a contraction over its columns
    /// ([`MmaData::as_lhs`]). A cmma fragment's cells lie where the instruction keeps them, and a
    /// register block is one unit's: both are drained into the plane's window and read from there.
    pub(crate) fn as_lhs<E: Numeric>(&self) -> PlaneTile<E> {
        match self {
            PlaneTile::Mma(fragment) => PlaneTile::new_Mma(fragment.as_lhs::<E>()),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) => panic!(
                "Tile::as_lhs: a manual-mma accumulator is read as the next contraction's factor \
                 in its registers; a cmma fragment or a register block drains into the plane's \
                 window first"
            ),
        }
    }

    /// Slices of this tile one unit keeps a factor for: a manual fragment's rows the unit holds
    /// cells of, every row of a cmma fragment or a register block.
    pub(crate) fn rows_held(&self) -> comptime_type!(usize) {
        match self {
            PlaneTile::Mma(fragment) => fragment.rows_held(),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) => {
                let (m, _) = self.shape();
                m
            }
        }
    }

    /// This tile carrying its partition's scratch. Only a cmma tile bounces.
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

    /// Add this tile's spilled cells into `mem`.
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

    /// The tile's `(m, n)`; only a cmma tile carries it.
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
            PlaneTile::Cmma(_) => panic!(
                "PlaneTile::scale: a cmma fragment is not read cell by cell, so a scale over one \
                 folds at the store instead"
            ),
            PlaneTile::Mma(d) => d.scale(factor),
            PlaneTile::Registers(d) => d.scale(factor),
        }
    }

    /// Fill this fragment from a memory `src`.
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
            PlaneTile::Registers(_) => {
                panic!("PlaneTile::load_window: a register accumulator is not a fill sink")
            }
        }
    }

    /// Store this tile into `mem`; `space` is the sink window's.
    pub(crate) fn store_window(&self, mem: &mut Memory<T>, #[comptime] space: Space) {
        let addressed = mem.store.addressed();
        match self {
            PlaneTile::Cmma(d) => {
                match comptime!(FragmentDrain::of(&mem.access, addressed, true)) {
                    FragmentDrain::Intrinsic => {
                        d.store_window(mem, comptime!(MatrixAxes::edges(&space).row_split))
                    }
                    FragmentDrain::Bounce => d.bounce_cast_window(mem, space),
                }
            }
            PlaneTile::Mma(d) => d.store_window(mem, space),
            // Same-type store; the block drains through `store_cast_window`.
            PlaneTile::Registers(d) => d.store_cast_window(mem, space),
        }
    }

    /// Add `src`'s block into this one, casting up. Only the register encoding promotes.
    pub(crate) fn add_cast_from<S: Numeric>(&mut self, src: &PlaneTile<S>) {
        match (self, src) {
            (PlaneTile::Registers(d), PlaneTile::Registers(s)) => d.add_cast_from(s),
            _ => panic!(
                "PlaneTile::add_cast_from: a register block promotes into a register block; a \
                 fragment accumulates in the element its instruction names"
            ),
        }
    }

    /// Store this tile into `mem`, casting; `space` is the sink window's.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        let addressed = mem.store.addressed();
        match self {
            PlaneTile::Cmma(d) => {
                let (m, n) = comptime!(d.shape);
                let in_place = casts_in_place::<T, Out>(m, n);
                match comptime!(FragmentDrain::of(&mem.access, addressed, in_place)) {
                    FragmentDrain::Intrinsic => {
                        d.store_cast_window(mem, comptime!(MatrixAxes::edges(&space).row_split))
                    }
                    FragmentDrain::Bounce => d.bounce_cast_window(mem, space),
                }
            }
            PlaneTile::Mma(d) => d.store_cast_window(mem, space),
            PlaneTile::Registers(d) => d.store_cast_window(mem, space),
        }
    }
}

/// The `m_tiles × n_tiles` grid of plane tiles one plane owns, row-major.
/// `Clone` duplicates the handles, not the tiles.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct PlanePartition<T: Numeric> {
    pub(crate) frags: Sequence<PlaneTile<T>>,
    #[cube(comptime)]
    pub m_tiles: usize,
    #[cube(comptime)]
    pub n_tiles: usize,
    /// One tile's edges along the window's trailing two axes.
    #[cube(comptime)]
    pub rows: usize,
    #[cube(comptime)]
    pub cols: usize,
    /// This plane's one-tile window of shared memory that a row-wise op bounces through.
    pub scratch: ComptimeOption<Shared<[T]>>,
    /// How much of this grid the scratch holds.
    #[cube(comptime)]
    pub held: Scratch,
}

#[cube]
impl<T: Numeric> PlanePartition<T> {
    /// The instruction this grid contracts through where it holds several fragments; `None` where
    /// it holds one.
    pub(crate) fn grid(&self) -> comptime_type!(Option<Instruction>) {
        let several = comptime!(self.m_tiles * self.n_tiles > 1);
        match self.frags.index(0usize) {
            PlaneTile::Cmma(_) => comptime!(several.then_some(Instruction::Cmma)),
            PlaneTile::Mma(d) => comptime!(several.then_some(Instruction::Mma { io: d.io })),
            PlaneTile::Registers(_) => comptime!(None),
        }
    }

    /// Whether this grid's tiles are cmma fragments, the one form that drains through a scratch.
    pub(crate) fn is_cmma(&self) -> comptime_type!(bool) {
        match self.frags.index(0usize) {
            PlaneTile::Cmma(_) => comptime!(true),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => comptime!(false),
        }
    }

    /// The `(mi, ni)` tile (a handle clone); comptime indices only.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> PlaneTile<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The one fragment of a `1 × 1` partition.
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

    /// The `sub_m × sub_n` block of fragments `step` selects, or one fragment if `1 × 1`.
    /// A level that cuts the partition must be walked with comptime coordinates.
    pub(crate) fn at_step(&self, step: &Step, #[comptime] space: Space) -> TileKind<T> {
        let edges = comptime!(MatrixAxes::edges(&space));
        let a0 = comptime!(space.axis_at(edges.row_split));
        let a1 = comptime!(space.axis_at(edges.col_split));
        // A single-tile axis folds to `0`; a `Dynamic` axis stays runtime (`None`).
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

    /// The `m_tiles × n_tiles` sub-partition at `(mi, ni)`, sharing the parent's tiles.
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

    /// This grid as the left factor of a contraction over its columns, fragment for fragment
    /// ([`PlaneTile::as_lhs`]).
    pub(crate) fn as_lhs<E: Numeric>(&self) -> PlanePartition<E> {
        let mut frags = Sequence::<PlaneTile<E>>::new();
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            frags.push(self.frags.index(i).as_lhs::<E>());
        }
        PlanePartition::<E> {
            frags,
            m_tiles: comptime!(self.m_tiles),
            n_tiles: comptime!(self.n_tiles),
            rows: comptime!(self.rows),
            cols: comptime!(self.cols),
            scratch: ComptimeOption::new_None(),
            held: Scratch::None,
        }
    }

    /// `self[r, :] *= corr[r]`, each tile bounced through the scratch.
    ///
    /// The scratch is the plane's own slot (`with_scratch` indexes it by `PLANE_POS`), so
    /// the bounce syncs the plane rather than the cube: every unit of the plane must call it,
    /// and the other planes need not.
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
        let units = PLANE_DIM as usize;
        let plane_unit = UNIT_POS_PLANE as usize;
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut tile = self.at(mi, ni);
                if moved {
                    tile.store_scratch(&scratch);
                }
                sync_plane();
                if moved {
                    let mut cell = plane_unit;
                    while cell < cells {
                        scratch[cell] *= corr[mi * m + cell / n];
                        cell += units;
                    }
                }
                sync_plane();
                if moved {
                    tile.load_scratch(&scratch);
                }
                sync_plane();
            }
        }
    }

    /// `self[i, :] *= factors[i]` for every slice `i` along the grid's columns, counted down the
    /// whole grid. Each tile is scaled where it is held ([`PlaneTile::mul_along`]), unless the
    /// scratch holds every fragment of the grid at once: then the plane spills them all, meets
    /// once, scales, meets once, and reloads them all.
    pub(crate) fn mul_along(&self, factors: &Array<T>) {
        // Only a cmma grid is opened with a scratch, so one that holds the whole grid is one.
        if comptime!(self.held == Scratch::WholeGrid) {
            self.mul_along_whole_grid(factors);
        } else {
            #[unroll]
            for mi in 0..self.m_tiles {
                #[unroll]
                for ni in 0..self.n_tiles {
                    let tile = self.at(mi, ni);
                    let held = tile.rows_held();
                    tile.mul_along(factors, comptime!(mi * held));
                }
            }
        }
    }

    /// [`mul_along`](Self::mul_along) over a grid of cmma fragments each with a slot of its own.
    fn mul_along_whole_grid(&self, factors: &Array<T>) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            self.cmma_at(i).spill_to_scratch();
        }
        sync_plane();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                self.cmma_at(comptime!(mi * self.n_tiles + ni))
                    .mul_spilled(factors, comptime!(mi * self.rows));
            }
        }
        sync_plane();
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            self.cmma_at(i).reload_from_scratch();
        }
        sync_plane();
    }

    /// The `i`-th tile of a grid of cmma fragments, in row-major order.
    fn cmma_at(&self, #[comptime] i: usize) -> CmmaData<T> {
        match self.frags.index(i).clone() {
            PlaneTile::Cmma(fragment) => fragment,
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => {
                panic!("PlanePartition: a grid holds one kind of tile")
            }
        }
    }

    /// An uninitialized plane-resident accumulator partition mirroring `space`'s grid.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mirror(
        #[comptime] space: Space,
        #[comptime] axes: MatrixAxes,
        #[comptime] form: Instruction,
        #[comptime] grid: GridShape,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] accumulation: Accumulation,
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
                    accumulation,
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
            // A `Dynamic` extent here is never read; fragments were sized from the statement.
            place: comptime!(Placement::new(space, depth, levels)),
        }
    }

    /// Uninitialized operand fragments for one region under `out`'s contraction, block-scaled
    /// where `block_scaled` says the region contracts through the device's block-scaled
    /// instruction ([`block_scales_here`]). `m`/`n` are the accumulator fragment's.
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
        #[comptime] block_scaled: bool,
    ) -> Tile<T> {
        let edges = comptime!(MatrixAxes::edges(&window));
        let a0 = comptime!(window.axis_at(edges.row_split));
        let a1 = comptime!(window.axis_at(edges.col_split));

        // The contracted axis is the one `out` lacks; `A` shares its rows, `B` its columns.
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
        let rows_along_free = comptime!(if ident == MatrixIdent::A { m } else { n });
        let k = comptime!(window.extent(contracted));
        let (e0, e1) = comptime!(if free == a0 {
            (rows_along_free, k)
        } else {
            (k, rows_along_free)
        });
        // A window listing the role's axes reversed (a weight `{n, k}`) loads col-major.
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
                frags.push(PlaneTile::<T>::operand(
                    form,
                    ident,
                    m,
                    n,
                    k,
                    layout,
                    block_scaled,
                ));
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

    /// This region of an operand, in the form `instruction` reads it: as it is where it already
    /// holds fragments or the form is a register block, else loaded into fragments contracting
    /// into `acc`.
    pub(crate) fn operand<Acc: Numeric>(
        src: &Tile<T>,
        acc: &Tile<Acc>,
        #[comptime] instruction: Instruction,
    ) -> Tile<T> {
        match &src.kind {
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => src.clone(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => match comptime!(instruction) {
                Instruction::Registers { .. } => src.clone(),
                Instruction::Cmma | Instruction::Mma { .. } => {
                    PlanePartition::<T>::fragments_in(src, acc, instruction)
                }
            },
        }
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
        let scaled = src.scaled();
        let packing = src.packing();
        let shared = src.is_shared();
        // The operand's own depth: the extent of the axes the accumulator lacks.
        let k = comptime!(
            src.place
                .space
                .axes()
                .filter(|&axis| !acc.place.space.contains(axis))
                .map(|axis| src.place.space.extent(axis))
                .product::<usize>()
        );
        let block_scaled = match comptime!(form) {
            Instruction::Mma { .. } => block_scales_here(src, m, n, k),
            Instruction::Cmma | Instruction::Registers { .. } => comptime!(false),
        };
        let mut frags = PlanePartition::<T>::store(
            comptime!(src.place.space.clone()),
            comptime!(form),
            comptime!(acc.place.space.clone()),
            comptime!(grid),
            comptime!(m),
            comptime!(n),
            comptime!(src.place.depth),
            comptime!(src.place.levels.clone()),
            block_scaled,
        );
        if comptime!(block_scaled) {
            // The instruction reads the stored values and their scales as they lie: nothing
            // decodes on the way, so each fragment loads its own window of the source.
            match &frags.kind {
                TileKind::PlanePartition(partition) => partition.fill_from(src),
                TileKind::Memory(_)
                | TileKind::PlaneTile(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("PlanePartition::fragments: operand fragments are a partition")
                }
            }
        } else if comptime!(scaled || packing != Packing::Plain || !shared) {
            // A scaled, packed or global operand loads from its plane's landing, as the direct
            // contraction does: a fragment reads a shared, plain window as it lies.
            let side = comptime!(Side::of(&src.place.space, &acc.place.space));
            let landing = src.landed(side, comptime!(acc.place.space.clone()));
            frags.copy_from(&landing);
            // The landing is this region's until every unit's load has read it.
            sync_plane();
        } else {
            frags.copy_from(src);
        }
        frags
    }

    /// Fill each tile from its window of `src`, in row-major order.
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

#[cube]
impl<E: Float> PlanePartition<E> {
    /// This grid's slices along its columns: a unit's register block where the grid is one, the
    /// fragments otherwise.
    pub(crate) fn slices(&self) -> AxisSlicesKind<E> {
        match self.at(0usize, 0usize) {
            PlaneTile::Registers(block) => {
                comptime!(assert!(
                    self.m_tiles * self.n_tiles == 1,
                    "Tile::along: a unit's slices are one register block"
                ));
                AxisSlicesKind::new_Registers(block)
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => AxisSlicesKind::new_Fragments(self.clone()),
        }
    }

    /// `self[r, c] = self[r, c] · scale + bias[r, c]` over every fragment, `bias` read at each
    /// cell's coordinates in the window over `space`.
    pub(crate) fn scale_add_along(&self, scale: E, bias: &Procedural<E>, #[comptime] space: Space) {
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                let mut fragment = self.at(mi, ni).readable();
                fragment.scale_add_rows(
                    scale,
                    bias,
                    comptime!(space.clone()),
                    comptime!((mi * self.rows, ni * self.cols)),
                );
            }
        }
    }

    /// Each row's max, starting from `seed`'s: every fragment's cells on it, then the units'.
    pub(crate) fn maxima_along(&self, seed: &Array<E>) -> Array<E> {
        let held = self.at(0usize, 0usize).rows_held();
        let mut maxima = Array::<E>::new(comptime!(self.m_tiles * held));
        #[unroll]
        for i in 0..comptime!(self.m_tiles * held) {
            maxima[i] = seed[i];
        }
        self.reduce_rows(&mut maxima, Monoid::Max);
        maxima
    }

    /// Each row's sum: every fragment's cells on it, then the units'.
    pub(crate) fn sums_along(&self) -> Array<E> {
        let held = self.at(0usize, 0usize).rows_held();
        let mut sums = Array::<E>::new(comptime!(self.m_tiles * held));
        #[unroll]
        for i in 0..comptime!(self.m_tiles * held) {
            sums[i] = E::from_int(0);
        }
        self.reduce_rows(&mut sums, Monoid::Sum);
        sums
    }

    /// `self[r, c] = exp(self[r, c] − rows[r])` over every fragment.
    pub(crate) fn exp_minus_along(&self, rows: &Array<E>) {
        let held = self.at(0usize, 0usize).rows_held();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                let mut fragment = self.at(mi, ni).readable();
                fragment.exp_minus_rows(rows, comptime!(mi * held));
            }
        }
    }

    /// `slots` combined under `monoid` with every fragment's cells on each row, the unit's own
    /// first and then across the units that share the row: one exchange a row, however many
    /// fragments it spans.
    fn reduce_rows(&self, slots: &mut Array<E>, #[comptime] monoid: Monoid) {
        let held = self.at(0usize, 0usize).rows_held();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                let fragment = self.at(mi, ni).readable();
                match comptime!(monoid) {
                    Monoid::Max => fragment.raise_row_maxima(slots, comptime!(mi * held)),
                    Monoid::Sum => fragment.add_row_sums(slots, comptime!(mi * held)),
                    Monoid::Prod | Monoid::Min => {
                        panic!("PlanePartition::reduce_rows: a softmax reduces by max and sum")
                    }
                }
            }
            self.at(mi, 0usize)
                .readable()
                .across_rows(slots, comptime!(mi * held), monoid);
        }
    }
}

#[cube]
impl<E: Float> PlaneTile<E> {
    /// This tile's slices along its columns: a register block's, or one fragment's.
    pub(crate) fn slices(&self) -> AxisSlicesKind<E> {
        match self {
            PlaneTile::Registers(block) => AxisSlicesKind::new_Registers(block.clone()),
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => AxisSlicesKind::new_Fragment(self.clone()),
        }
    }

    /// This fragment where its cells are read by rows: a manual fragment's registers. A cmma
    /// fragment's cells lie where the instruction keeps them, unknown to the program, and are
    /// read after draining it into its plane's window.
    pub(crate) fn readable(&self) -> MmaData<E> {
        match self {
            PlaneTile::Mma(fragment) => fragment.clone(),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) => panic!(
                "PlaneTile: a cmma fragment's slices are read after draining it into its plane's \
                 window; a manual-mma fragment's are read in place"
            ),
        }
    }
}

/// The never-walked level cutting an operand's window into the partition's fragments.
fn fragment_level(window: &Space, frag: (usize, usize), tiles: (usize, usize)) -> Level {
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
