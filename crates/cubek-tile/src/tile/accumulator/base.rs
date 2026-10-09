//! The plane-resident accumulator an output opens ([`Accumulate`]).

use cubecl::prelude::*;

use crate::ops::matmul::leaf::memory;
use crate::*;

/// The shape a plane-resident accumulator is opened at: `tiles` per plane, each `m × n`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GridShape {
    pub(crate) tiles: (usize, usize),
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl GridShape {
    /// The grid an accumulator at `place` contracting `lhs` holds.
    pub(crate) fn new(place: &Placement, lhs: &Space) -> Self {
        let levels = place.below();
        let tiles = partition_shape(&place.space, levels);
        let leaf = place.space.leaf(levels);
        let lhs_leaf = lhs.leaf(levels);
        let axes = MatrixAxes::accumulator(&leaf, &lhs_leaf);
        // Use the accumulator's own edges: a split column group is one edge.
        GridShape {
            tiles,
            m: axes.rows(&leaf),
            n: axes.cols(&leaf),
            k: lhs_leaf.contracted_extent(&place.space),
        }
    }
}

/// How much of a plane's accumulator its scratch holds at once. Serialized names are persisted in
/// autotune keys and must not change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Scratch {
    /// No window: fragments store through their intrinsic, which cannot add; a drain that needs
    /// one opens [`OneTile`](Scratch::OneTile).
    None,
    /// One tile, bounced through shared memory with three cube-wide barriers each.
    OneTile,
    /// The plane's whole grid, two barriers for the whole drain.
    #[serde(rename = "WholePartition")]
    WholeGrid,
}

/// The persisted setting's spelling.
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

    /// The slot tile `i` of a grid of `tiles` spills into.
    pub(crate) fn slot_of(self, i: usize, tiles: usize) -> usize {
        match self {
            Scratch::WholeGrid => i,
            Scratch::None | Scratch::OneTile => {
                let _ = (i, tiles);
                0
            }
        }
    }

    /// Whether a fragment goes through shared memory on its way out; forced when folding.
    pub fn bounces(self) -> bool {
        !matches!(self, Scratch::None)
    }

    /// This setting raised to what `folds` demands.
    pub fn at_least_bouncing(self, folds: bool) -> Scratch {
        match (folds, self) {
            (true, Scratch::None) => Scratch::OneTile,
            (_, scratch) => scratch,
        }
    }
}

/// The planes `levels` distribute `space` across; panics on a runtime-only count.
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

/// What an output does with the sum contracted into it: open, scratch and drain an accumulator.
#[cube]
pub trait Accumulate<Acc: Numeric>: CubeType + Sized {
    /// What one plane sums `lhs · rhs` into under `semiring`, in the form `instruction` names,
    /// opened at the semiring's identity.
    fn accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] instruction: Instruction,
        #[comptime] semiring: Semiring,
    ) -> Tile<EA>;

    /// A register block over this tile's grid for a reduction over `input` under `monoid`,
    /// opened at its identity.
    fn block_reducer<EA: Numeric, In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// This accumulator opened with a per-plane shared-memory scratch for bouncing drains.
    /// [`drained_into`](Self::drained_into) opens the smallest one itself where it needs one;
    /// stating it chooses how much of the grid it holds, or serves a drain region by region.
    fn with_scratch(self, #[comptime] scratch: Scratch) -> Self;

    /// Drain this accumulator into `dest`, cast to its element. `self` and `dest` must be
    /// indexed by the same region. A cmma grid opened with no scratch opens one where `dest`
    /// needs its fragments bounced.
    fn drained_into<Out: Numeric>(&self, dest: &Tile<Out>);
}

#[cube]
impl<Acc: Numeric> Accumulate<Acc> for Tile<Acc> {
    fn accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] instruction: Instruction,
        #[comptime] semiring: Semiring,
    ) -> Tile<EA> {
        match comptime!(instruction) {
            Instruction::Registers { config } => {
                register_accumulator::<Acc, EA, EL, ER>(self, lhs, rhs, config, semiring)
            }
            Instruction::Cmma | Instruction::Mma { .. } => {
                let vector_size = self.vector_size();
                accumulator_in::<Acc, EA, EL>(
                    self,
                    lhs,
                    instruction,
                    vector_size,
                    1usize,
                    comptime!(Accumulation::Contraction(semiring)),
                )
            }
        }
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
            comptime!(Accumulation::Reduction(monoid)),
        )
    }

    fn with_scratch(self, #[comptime] scratch: Scratch) -> Tile<Acc> {
        match &self.kind {
            TileKind::PlanePartition(p) => {
                let (m, n) = p.at(0usize, 0usize).shape();
                let cells = comptime!(m * n);
                let tiles = comptime!(p.m_tiles * p.n_tiles);
                let slots = comptime!(scratch.slots(tiles));
                // Indexed by hardware plane position, not level position: the launch decides the cube's shape.
                let planes = comptime!(plane_windows(&self.place.space, &self.place.levels));
                let plane = PLANE_POS.cast::<usize>() * comptime!(cells * slots);
                let shared = Shared::<[Acc]>::new_slice(comptime!(cells * slots * planes));
                // Each fragment carries its own slot, so drain order doesn't matter.
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
        // A destination the fragments cannot store to directly needs a scratch: where none was
        // opened, the drain opens the smallest.
        let unopened = self.scratch_unopened();
        let in_place = self.casts_in_place_to::<Out>();
        let bounces = dest.fragments_bounce(in_place);
        if comptime!(unopened && bounces) {
            let opened = self.clone().with_scratch(Scratch::OneTile);
            opened.drained_into(dest);
        } else {
            self.drain_as_opened(dest);
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// This plane-resident accumulator as the left factor of the next contraction, the one over
    /// its columns, its cells cast to `E` where its units hold them: what an attention contracts
    /// its probabilities with the values through, without a round trip through shared memory.
    ///
    /// Only a manual-mma accumulator is read in place, whose layout is the `A` fragment's of an
    /// instruction as deep as the accumulator is wide; any other drains into its plane's window
    /// and is staged from there.
    pub fn as_lhs<E: Numeric>(&self) -> Tile<E> {
        let place = comptime!(self.place.clone());
        match &self.kind {
            TileKind::PlanePartition(partition) => {
                Tile::new(TileKind::new_PlanePartition(partition.as_lhs::<E>()), place)
            }
            TileKind::PlaneTile(tile) => {
                Tile::new(TileKind::new_PlaneTile(tile.as_lhs::<E>()), place)
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::as_lhs: only a plane-resident accumulator is a contraction's output")
            }
        }
    }

    /// Whether this accumulator's fragments cast to `Out` are fragments the device holds
    /// ([`casts_in_place`]): what decides whether a cmma grid draining into an `Out` destination
    /// stores whole or bounces.
    fn casts_in_place_to<Out: Numeric>(&self) -> comptime_type!(bool) {
        match &self.kind {
            // The fragment's own edges: the placement's space spans every plane's grid.
            TileKind::PlanePartition(p) => {
                casts_in_place::<Acc, Out>(comptime!(p.rows), comptime!(p.cols))
            }
            TileKind::PlaneTile(_) => {
                let (_, m, n) = self.fragment_grid();
                casts_in_place::<Acc, Out>(m, n)
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(true),
        }
    }

    /// Whether this is a grid of cmma fragments opened with no scratch.
    fn scratch_unopened(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::PlanePartition(p) => {
                let cmma = p.is_cmma();
                comptime!(cmma && p.held == Scratch::None)
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// [`drained_into`](Accumulate::drained_into) for the fragments of chunk `turn` alone, of
    /// `chunks` the fragment grid is dealt into by place: what one writer of a cyclic schedule
    /// lands in one round, each fragment drained in exactly one of the `chunks` rounds. It reaches
    /// every barrier a whole drain does, so writers landing different chunks meet at all of them.
    /// A grid opened with no scratch drains through the smallest, as a whole drain does.
    pub(crate) fn drained_chunk_into<Out: Numeric>(
        &self,
        dest: &Tile<Out>,
        turn: usize,
        #[comptime] chunks: usize,
    ) {
        let unopened = self.scratch_unopened();
        let in_place = self.casts_in_place_to::<Out>();
        let bounces = dest.fragments_bounce(in_place);
        if comptime!(unopened && bounces) {
            let opened = self.clone().with_scratch(Scratch::OneTile);
            opened.drained_chunk_into(dest, turn, chunks);
        } else {
            self.drain_chunk_as_opened(dest, turn, chunks);
        }
    }

    /// [`drained_chunk_into`](Tile::drained_chunk_into) through the scratch this grid was opened
    /// with.
    fn drain_chunk_as_opened<Out: Numeric>(
        &self,
        dest: &Tile<Out>,
        turn: usize,
        #[comptime] chunks: usize,
    ) {
        match &self.kind {
            TileKind::PlanePartition(p) => match comptime!(DrainPlan::new(p.held)) {
                DrainPlan::Straight => drain_chunk_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Copy),
                    0usize,
                    turn,
                    chunks,
                ),
                DrainPlan::BounceEach => drain_chunk_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Bounce),
                    0usize,
                    turn,
                    chunks,
                ),
                DrainPlan::BounceTogether => {
                    sync_cube();
                    drain_chunk_below::<Acc, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Spill),
                        0usize,
                        turn,
                        chunks,
                    );
                    sync_cube();
                    drain_chunk_below::<Acc, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Add),
                        0usize,
                        turn,
                        chunks,
                    );
                    sync_cube();
                }
            },
            // One tile: it is the one chunk of its grid. A fragment bouncing into its destination
            // meets the cube at its barriers whatever the turn, so it bounces through the
            // predicated pass; anything else stores with no barrier at all.
            TileKind::PlaneTile(t) => {
                let cmma = match t {
                    PlaneTile::Cmma(_) => comptime!(true),
                    PlaneTile::Mma(_) | PlaneTile::Registers(_) => comptime!(false),
                };
                let in_place = self.casts_in_place_to::<Out>();
                let bounces = dest.fragments_bounce(in_place);
                if comptime!(cmma && bounces) {
                    drain_chunk_leaf::<Acc, Out>(
                        self,
                        dest,
                        comptime!(DrainPass::Bounce),
                        turn == 0,
                    );
                } else {
                    drain_chunk_leaf::<Acc, Out>(self, dest, comptime!(DrainPass::Copy), turn == 0);
                }
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::drained_chunk_into: a plane-resident accumulator drains; nothing else does"
                )
            }
        }
    }

    /// [`drained_into`](Accumulate::drained_into) through the scratch this grid was opened with.
    fn drain_as_opened<Out: Numeric>(&self, dest: &Tile<Out>) {
        match &self.kind {
            TileKind::PlanePartition(p) => match comptime!(DrainPlan::new(p.held)) {
                DrainPlan::Straight => drain_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Copy),
                ),
                // One slot shared: each spill and add pair off at its leaf.
                DrainPlan::BounceEach => drain_below::<Acc, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Bounce),
                ),
                // Own slots: every spill precedes any add, so the descent runs twice.
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
            // No grid: the accumulator is one tile and the drain is one store.
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

/// A register block `out`'s plane sums `lhs · rhs` into, run under `config`.
#[cube]
fn register_accumulator<Acc: Numeric, EA: Numeric, EL: Numeric, ER: Numeric>(
    out: &Tile<Acc>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) -> Tile<EA> {
    let lw = lhs.vector_size();
    // The block's lines are the rhs's loads, or, where the rhs runs along the contraction, one
    // column's run of a load summed into a cell.
    let rhs_load = rhs.vector_tile();
    let rw = comptime!(rhs_load.run_length());
    let aw = out.vector_size();
    let fold = comptime!(memory::contracted_per_step(
        &lhs.place.space,
        &rhs.place.space,
        &out.place.space,
        lw,
        rw,
        aw
    ));
    // A block outliving the leaf cannot spread lines, so its lines and the sink's agree or fold.
    comptime!(assert!(
        fold > 1 || rw == aw,
        "Tile::accumulator: the block's lines are the rhs's ({rw} wide) and drain into \
         {aw}-wide cells; a stage served wider than its sink is the memory-backed leaf's \
         (Tile::accumulating)"
    ));
    // A block contracting runs along the contraction sums each into its cell, one value wide.
    let width = comptime!(if fold > 1 { 1 } else { rw });
    accumulator_in::<Acc, EA, EL>(
        out,
        lhs,
        comptime!(Instruction::Registers { config }),
        width,
        fold,
        comptime!(Accumulation::Contraction(semiring)),
    )
}

/// The plane-resident grid an accumulator gathers in, in `form`, at its identity.
#[cube]
fn accumulator_in<Acc: Numeric, EA: Numeric, EL: Numeric>(
    out: &Tile<Acc>,
    lhs: &Tile<EL>,
    #[comptime] form: Instruction,
    #[comptime] vector_size: usize,
    #[comptime] fold: usize,
    #[comptime] accumulation: Accumulation,
) -> Tile<EA> {
    comptime!(assert!(
        matches!(form, Instruction::Registers { .. })
            || accumulation == Accumulation::Contraction(Semiring::SUM_PROD),
        "Tile::accumulator: a hardware instruction accumulates under the sum of products alone, \
         not {accumulation:?}; contract in a register block to fold under another"
    ));
    let mut acc = PlanePartition::<EA>::mirror(
        comptime!(out.place.space.clone()),
        comptime!(MatrixAxes::accumulator(&out.place.space, &lhs.place.space)),
        comptime!(form),
        comptime!(GridShape::new(&out.place, &lhs.place.space)),
        vector_size,
        fold,
        accumulation,
        comptime!(out.place.depth),
        comptime!(out.place.levels.clone()),
    );
    acc.init_identity(comptime!(accumulation.monoid()));
    acc
}

#[cube]
impl<EA: Numeric> Tile<EA> {
    /// A plane-resident accumulator over one region of `walk` along `axes`, summing `lhs` times a
    /// factor in the matrix instruction `instruction` names, opened at zero: the accumulator
    /// [`Tile::scratch`] would open, without the memory, for a product contracted again before
    /// anything drains it, as an attention's score.
    pub fn stage_accumulator<EL: Numeric>(
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        lhs: &Tile<EL>,
        #[comptime] instruction: Instruction,
    ) -> Tile<EA> {
        comptime!(assert!(
            !matches!(instruction, Instruction::Registers { .. }),
            "Tile::stage_accumulator: a register block's accumulator is a unit's, opened on the \
             output it drains into"
        ));
        let place = comptime!(Placement::new(
            walk.level.child(&walk.space).subspace(&axes),
            walk.depth(),
            walk.parent.path.root_levels(),
        ));
        let accumulation = comptime!(Accumulation::Contraction(Semiring::SUM_PROD));
        let mut acc = PlanePartition::<EA>::mirror(
            comptime!(place.space.clone()),
            comptime!(MatrixAxes::accumulator(&place.space, &lhs.place.space)),
            instruction,
            comptime!(GridShape::new(&place, &lhs.place.space)),
            1usize,
            1usize,
            accumulation,
            comptime!(place.depth),
            comptime!(place.levels.clone()),
        );
        acc.init_identity(comptime!(accumulation.monoid()));
        acc
    }
}

/// [`Accumulate::drained_into`]'s descent over `levels[i..]`, applying `pass` at every leaf.
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
        // A walked level's coordinates must fold to constants: its regions select fragments.
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

/// [`Tile::drained_chunk_into`]'s descent over `levels[i..]`: [`drain_below`]'s, the fragment
/// grid's coordinates summed along the way into `place`, the chunk a fragment falls in being
/// `place` modulo `chunks`.
#[cube]
#[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
fn drain_chunk_below<Acc: Numeric, Out: Numeric>(
    acc: &Tile<Acc>,
    dest: &Tile<Out>,
    #[comptime] levels: Vec<Level>,
    #[comptime] i: usize,
    #[comptime] pass: DrainPass,
    place: usize,
    turn: usize,
    #[comptime] chunks: usize,
) {
    if comptime!(i == levels.len()) {
        drain_chunk_leaf::<Acc, Out>(acc, dest, pass, place % chunks == turn);
    } else {
        let level = comptime!(levels[i].clone());
        let axes = comptime!(dest.place.space.axes().collect::<Vec<_>>());
        // Only the fragment grid, a walked level, places a fragment: every plane holding the same
        // tile reaches its fragments at the same places, the planes' own level adding nothing.
        if comptime!(level.coverage() == Coverage::Walk) {
            for region in dest.over(&level).unrolled() {
                let mut here = place;
                #[unroll]
                for a in 0..comptime!(axes.len()) {
                    here += region.coord(comptime!(axes[a]));
                }
                drain_chunk_below::<Acc, Out>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                    here,
                    turn,
                    chunks,
                );
            }
        } else {
            for region in dest.over(&level) {
                drain_chunk_below::<Acc, Out>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                    place,
                    turn,
                    chunks,
                );
            }
        }
    }
}

/// [`drain_leaf`] for a fragment that lands only where `lands`: every barrier is reached
/// whatever it says, so planes landing different fragments still meet at the same ones.
#[cube]
fn drain_chunk_leaf<Acc: Numeric, Out: Numeric>(
    acc: &Tile<Acc>,
    dest: &Tile<Out>,
    #[comptime] pass: DrainPass,
    lands: bool,
) {
    match comptime!(pass) {
        DrainPass::Copy => {
            if lands {
                let mut window = dest.clone();
                window.copy_cast_from(acc);
            }
        }
        DrainPass::Spill => {
            if lands {
                acc.spill_to_scratch();
            }
        }
        DrainPass::Add => {
            if lands {
                let mut window = dest.clone();
                window.add_from_scratch(acc);
            }
        }
        DrainPass::Bounce => {
            let mut window = dest.clone();
            sync_cube();
            if lands {
                acc.spill_to_scratch();
            }
            sync_cube();
            if lands {
                window.add_from_scratch(acc);
            }
            sync_cube();
        }
    }
}

/// One tile of a drain against the one window it lands in.
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
