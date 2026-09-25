//! The plane-resident accumulator an output opens ([`Accumulate`]): a grid of plane tiles
//! mirroring the output's own grid under the levels below it, contracted into, and drained back
//! one tile per cell.
//!
//! ```ignore
//! let mut acc = c.accumulator::<EA, EL, ER>(&a, &b, instruction, Monoid::Sum);
//! acc.zero();
//! for step in plane.walk() { /* acc.at(&cell).mma(..) */ }
//! acc.drained_into(&c);
//! ```

use cubecl::prelude::*;

use crate::ops::matmul::leaf::memory;
use crate::*;

/// The shape a plane-resident accumulator is opened at: the `tiles` one plane holds, each `m × n`
/// and contracting `k` a step. Read off the accumulator's placement (the walked levels below it
/// cut its grid) and the lhs it is contracted with (which sizes `k`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GridShape {
    pub(crate) tiles: (usize, usize),
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl GridShape {
    /// The grid an accumulator at `place` contracting `lhs` holds: the walked levels below its
    /// depth cut its tiles, the leaf they reach is one tile, and `lhs`'s leaf contracts `k`.
    pub(crate) fn new(place: &Placement, lhs: &Space) -> Self {
        let levels = place.below();
        let tiles = partition_shape(&place.space, levels);
        let leaf = place.space.leaf(levels);
        let lhs_leaf = lhs.leaf(levels);
        let axes = MatrixAxes::accumulator(&leaf, &lhs_leaf);
        // The edges the accumulator's own axes give, not its last two: a split column group is
        // one edge, and sizing the block off the innermost axis alone would cut it in half.
        GridShape {
            tiles,
            m: axes.rows(&leaf),
            n: axes.cols(&leaf),
            k: lhs_leaf.contracted_extent(&place.space),
        }
    }
}

/// How much of a plane's accumulator its scratch holds at once.
///
/// A fragment's cells sit across the plane's lanes in a layout only the hardware knows, so
/// anything that must touch them one at a time spills the tile to shared memory first. How much is
/// resident is a trade and not a fact: barriers against bytes.
///
/// **There is no size between the two.** A middle size would hand each tile a slot chosen by its
/// drain, whose walk order and the grid's index order nothing makes agree. At these two sizes a
/// tile's slot is a fact about the tile: its own index, or the one slot there is.
///
/// Serialized because a caller's setting rides a persisted autotune key, so the value a winner was
/// measured at has to come back; the on-disk spellings are the ones it was measured under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Scratch {
    /// No window. A fragment stores straight to memory through its own intrinsic, which is the
    /// cheapest drain there is and the only one that cannot add: a folding destination refuses.
    None,
    /// One tile. A fragment bounces through shared memory, one at a time, three cube-wide
    /// barriers each. The smallest window that lets a drain touch a fragment's cells.
    OneTile,
    /// The plane's whole grid. Two barriers for the whole drain, at as many times the footprint
    /// as the grid has tiles.
    #[serde(rename = "WholePartition")]
    WholeGrid,
}

/// `none`, `one_tile`, `whole_partition`: what a persisted setting is read by.
impl core::fmt::Display for Scratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Scratch::None => write!(f, "none"),
            Scratch::OneTile => write!(f, "one_tile"),
            Scratch::WholeGrid => write!(f, "whole_partition"),
        }
    }
}

impl Scratch {
    /// Slots the window holds for a grid of `tiles`.
    pub fn slots(self, tiles: usize) -> usize {
        match self {
            Scratch::None => 0,
            Scratch::OneTile => 1,
            Scratch::WholeGrid => tiles,
        }
    }

    /// The slot tile `i` of a grid of `tiles` spills into: its own where every tile has one, the
    /// only slot otherwise.
    pub fn slot_of(self, i: usize, tiles: usize) -> usize {
        match self {
            Scratch::WholeGrid => i,
            Scratch::None | Scratch::OneTile => {
                let _ = (i, tiles);
                0
            }
        }
    }

    /// Whether a fragment goes through shared memory on its way out at all.
    ///
    /// **Forced for a destination that folds**: the intrinsic's store overwrites, so the cells must
    /// become addressable before they can be added. A replacing destination may want it too, since
    /// a bounced drain writes lines the lanes distribute between them; which is faster is a measurement.
    pub fn bounces(self) -> bool {
        !matches!(self, Scratch::None)
    }

    /// This setting raised to what `folds` demands: a folding destination cannot be drained
    /// without a window, so [`None`](Self::None) is not a value it admits.
    pub fn at_least_bouncing(self, folds: bool) -> Scratch {
        match (folds, self) {
            (true, Scratch::None) => Scratch::OneTile,
            (_, scratch) => scratch,
        }
    }
}

/// The planes `levels` distribute `space` across: the product, over every level distributed on the cube's
/// planes, of the instances its plane-distributed axes take. One where no level rides the planes.
///
/// Each level's count is read against the space its parents hand it, so a count that is
/// only known at runtime is refused here rather than read as one.
pub(crate) fn plane_windows(space: &Space, levels: &[Level]) -> usize {
    let mut handed = space.clone();
    let mut planes = 1;
    for level in levels {
        for axis in level.axes() {
            if level.coverage() == Coverage::Distribute(ComputeScope::Plane)
                && level.distributes(axis)
            {
                planes *= level.instances_along(&handed, axis).unwrap_or_else(|| {
                    panic!(
                        "Tile::with_scratch: {axis:?} is distributed across the cube's planes at a \
                         count only the launch knows, so the plane's windows cannot be sized"
                    )
                });
            }
        }
        handed = level.child(&handed);
    }
    planes
}

/// What an output does with the sum contracted into it: open a plane-resident accumulator over
/// its own grid, give it the scratch a bouncing drain needs, and take the sum back.
///
/// Implemented once for [`Tile`]: the accumulator's shape is the tile's placement (the walked
/// levels below it cut its grid) and the operands' widths; nothing is restated.
#[cube]
pub trait Accumulate<Acc: Numeric>: CubeType + Sized {
    /// What one plane sums into, in the form `instruction` names.
    ///
    /// **The one opener a kernel whose instruction is data wants.** A derivation electing a form
    /// from what the device offers hands it here; the kernel never matches on it. The register
    /// block reads both operands to line its cells; the fragment forms read the lhs alone, for `k`.
    fn accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] instruction: Instruction,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// The plane-resident accumulator this output contracts in through the tensor-core
    /// instruction: an uninitialized grid of cmma fragments mirroring this tile's grid, drained
    /// after the walk one fragment per cell ([`drained_into`](Self::drained_into)). `lhs` sizes
    /// `k`.
    fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`cmma_accumulator`](Self::cmma_accumulator) through the manual-mma instruction, whose
    /// fragment transports are `io`'s.
    fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] io: MmaIo,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`cmma_accumulator`](Self::cmma_accumulator) through the software instruction: a register
    /// block per tile of the grid, run under `config`.
    ///
    /// The block's lines are the rhs's, so it reads both operands where the hardware forms read
    /// the lhs alone. An rhs lined along the accumulator gives lines of neighbouring cells; one
    /// lined along `k` gives each cell a line of partials folded on drain, and the tile is scalar.
    fn block_accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`block_accumulator`](Self::block_accumulator) for a reduction, whose one operand is
    /// `input`: the block's lines are this tile's, there being no rhs to line them by.
    fn block_reducer<EA: Numeric, In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// This plane-resident accumulator opened with a scratch: a window of shared memory per plane
    /// of the cube that a fragment bounces through where its cells must be touched one at a time.
    /// Stated where the accumulator opens: the scratch is its residence. How many planes share
    /// the cube is read off the partitioning; the plane's width is the launch's.
    fn with_scratch(self, #[comptime] scratch: Scratch) -> Self;

    /// Drain this plane-resident accumulator into `dest`, cast to `dest`'s element, one tile per
    /// cell of its grid.
    ///
    /// **`self` and `dest` are indexed by the same region**, so a caller that narrows one narrows
    /// the other to the same window. An accumulator opened wider than the destination handed over
    /// resolves each region somewhere else and drains the wrong cells.
    ///
    /// **The scratch's size decides the barrier count** ([`Scratch`]). With one tile resident each
    /// tile drains on its own, three cube-wide barriers apiece. With the whole grid resident no
    /// two tiles share a slot: every spill, one barrier, every add, two in all.
    fn drained_into<Out: Numeric>(&self, dest: &Tile<Out>);
}

#[cube]
impl<Acc: Numeric> Accumulate<Acc> for Tile<Acc> {
    fn accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] instruction: Instruction,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        match comptime!(instruction) {
            Instruction::Registers { config } => {
                self.block_accumulator::<EA, EL, ER>(lhs, rhs, config, monoid)
            }
            Instruction::Cmma => self.cmma_accumulator::<EA, EL>(lhs, monoid),
            Instruction::Mma { io } => self.mma_accumulator::<EA, EL>(lhs, io, monoid),
        }
    }

    fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        accumulator_in::<Acc, EA, EL>(
            self,
            lhs,
            comptime!(Instruction::Cmma),
            vector_size,
            1usize,
            monoid,
        )
    }

    fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] io: MmaIo,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        accumulator_in::<Acc, EA, EL>(
            self,
            lhs,
            comptime!(Instruction::Mma { io }),
            vector_size,
            1usize,
            monoid,
        )
    }

    fn block_accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let lw = lhs.vector_size();
        let rw = rhs.vector_size();
        let aw = self.vector_size();
        let fold = comptime!(memory::contracted_per_step(
            &lhs.place.space,
            &rhs.place.space,
            &self.place.space,
            lw,
            rw,
            aw
        ));
        // A memory leaf spreads a line wider than its scalar sink across cells; a block that
        // outlives the leaf has no such step, so its lines and the sink's agree or fold.
        comptime!(assert!(
            fold > 1 || rw == aw,
            "Tile::block_accumulator: the block's lines are the rhs's ({rw} wide) and drain into \
             {aw}-wide cells; a stage served wider than its sink is the memory-backed leaf's \
             (Tile::mma_with)"
        ));
        accumulator_in::<Acc, EA, EL>(
            self,
            lhs,
            comptime!(Instruction::Registers { config }),
            rw,
            fold,
            monoid,
        )
    }

    fn block_reducer<EA: Numeric, In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        let vector_size = self.vector_size();
        accumulator_in::<Acc, EA, In>(
            self,
            input,
            comptime!(Instruction::Registers { config }),
            vector_size,
            1usize,
            monoid,
        )
    }

    fn with_scratch(self, #[comptime] scratch: Scratch) -> Tile<Acc> {
        match &self.kind {
            TileKind::PlanePartition(p) => {
                let (m, n) = p.at(0usize, 0usize).shape();
                let cells = comptime!(m * n);
                let tiles = comptime!(p.m_tiles * p.n_tiles);
                let slots = comptime!(scratch.slots(tiles));
                // One window per plane of the cube, this plane's found by the hardware's own
                // plane index; the cube's whole claim sits behind it. Not the level's position:
                // the launch decides the cube's shape, and a team wider than a plane puts two
                // planes on one of its rows.
                let planes = comptime!(plane_windows(&self.place.space, &self.place.levels));
                let plane = PLANE_POS.cast::<usize>() * comptime!(cells * slots);
                let shared = Shared::<[Acc]>::new_slice(comptime!(cells * slots * planes));
                // **Every fragment carries its own slot.** One taken off the grid by `at` then
                // bounces on its own, and, where the whole grid is resident, no two fragments
                // share a slot, so nothing depends on the order a drain walks them in.
                let mut frags = Sequence::<PlaneTile<Acc>>::new();
                #[unroll]
                for i in 0..tiles {
                    let start = plane + comptime!(cells * scratch.slot_of(i, tiles));
                    let slot = shared.clone().map(|s| &s[start..start + cells]);
                    frags.push(p.frags.index(i).clone().with_scratch(slot));
                }
                Tile::new(
                    TileKind::new_PlanePartition(PlanePartition::<Acc> {
                        frags,
                        m_tiles: comptime!(p.m_tiles),
                        n_tiles: comptime!(p.n_tiles),
                        rows: comptime!(p.rows),
                        cols: comptime!(p.cols),
                        scratch: ComptimeOption::new_Some(
                            shared.clone().map(|s| &s[plane..plane + cells]),
                        ),
                        held: comptime!(scratch),
                    }),
                    comptime!(self.place.clone()),
                )
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::with_scratch: a scratch backs a plane-resident accumulator")
            }
        }
    }

    fn drained_into<Out: Numeric>(&self, dest: &Tile<Out>) {
        match &self.kind {
            // A grid below the open: the descent walks the levels it was cut by, narrowing both
            // sides, and lands one tile against one window.
            TileKind::PlanePartition(p) => match comptime!(DrainPlan::new(p.held)) {
                DrainPlan::Straight => drain_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Copy),
                ),
                // One slot between the tiles, so each one's spill and add pair off at its leaf.
                DrainPlan::BounceEach => drain_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Bounce),
                ),
                // Every tile has its own slot, so every spill happens before any add: the
                // descent runs twice and the barriers are the whole drain's.
                DrainPlan::BounceTogether => {
                    sync_cube();
                    drain_below::<Acc, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Spill),
                    );
                    sync_cube();
                    drain_below::<Acc, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Add),
                    );
                    sync_cube();
                }
            },
            // No grid: the accumulator is one tile and the drain is one store. The register
            // leaves are this shape, since their block is the plane's whole box rather than a
            // grid of it.
            TileKind::PlaneTile(_) => {
                drain_leaf::<Acc, Out>(self, dest, comptime!(DrainPass::Copy))
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::drained_into: a plane-resident accumulator drains; nothing else does")
            }
        }
    }
}

/// The plane-resident grid an accumulator contracts in, in `form`, uninitialized and shaped to
/// meet `lhs` at the instruction. `vector_size` is its lines' width and `fold` what a line holds
/// ([`RegisterData::fold`]); only the software form reads them.
#[cube]
fn accumulator_in<Acc: Numeric, EA: Numeric, EL: Numeric>(
    out: &Tile<Acc>,
    lhs: &Tile<EL>,
    #[comptime] form: Instruction,
    #[comptime] vector_size: usize,
    #[comptime] fold: usize,
    #[comptime] monoid: Monoid,
) -> Tile<EA> {
    PlanePartition::<EA>::mirror(
        comptime!(out.place.space.clone()),
        comptime!(MatrixAxes::accumulator(&out.place.space, &lhs.place.space)),
        comptime!(form),
        comptime!(GridShape::new(&out.place, &lhs.place.space)),
        vector_size,
        fold,
        monoid,
        comptime!(out.place.depth),
        comptime!(out.place.levels.clone()),
    )
}

/// [`Accumulate::drained_into`]'s descent: `levels[i..]` walked in the destination's own axes,
/// narrowing both sides one level at a time, `pass` applied to every leaf of it.
///
/// **The levels the grid was read off are the levels the drain walks** ([`GridShape::new`]), which
/// is what makes the leaf one tile: a level that cuts the grid hands out one region per tile, a
/// level that distributes hands out this instance's own window, and a level the destination does not
/// span hands out one region and narrows nothing. A contraction the accumulator outlives is of
/// that last kind, so it is walked once here however many steps it takes.
#[cube]
fn drain_below<Acc: Numeric, Out: Numeric>(
    acc: &Tile<Acc>,
    dest: &Tile<Out>,
    #[comptime] levels: Vec<Level>,
    #[comptime] i: usize,
    #[comptime] pass: DrainPass,
) {
    if comptime!(i == levels.len()) {
        drain_leaf::<Acc, Out>(acc, dest, pass);
    } else {
        let level = comptime!(levels[i].clone());
        // A walked level's regions select fragments, so its coordinates must fold to constants;
        // a distributed one selects nothing and is one region whatever its count.
        if comptime!(level.coverage() == Coverage::Walk) {
            for region in dest.over(&level).unrolled() {
                drain_below::<Acc, Out>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                );
            }
        } else {
            for region in dest.over(&level) {
                drain_below::<Acc, Out>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                );
            }
        }
    }
}

/// One tile of a drain, against the one window it lands in: what every leaf of the descent runs.
#[cube]
fn drain_leaf<Acc: Numeric, Out: Numeric>(
    acc: &Tile<Acc>,
    dest: &Tile<Out>,
    #[comptime] pass: DrainPass,
) {
    match comptime!(pass) {
        DrainPass::Copy => {
            let mut window = dest.clone();
            window.copy_cast_from(acc);
        }
        DrainPass::Spill => acc.spill_to_scratch(),
        DrainPass::Add => {
            let mut window = dest.clone();
            window.add_from_scratch(acc);
        }
        DrainPass::Bounce => {
            let mut window = dest.clone();
            sync_cube();
            acc.spill_to_scratch();
            sync_cube();
            window.add_from_scratch(acc);
            sync_cube();
        }
    }
}
