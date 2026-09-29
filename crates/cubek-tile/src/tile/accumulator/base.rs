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
    /// No window: fragments store through their intrinsic, which cannot add.
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
    /// What one plane sums into, in the form `instruction` names.
    fn accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] instruction: Instruction,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// A cmma-fragment accumulator mirroring this tile's grid; `lhs` sizes `k`.
    fn cmma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`cmma_accumulator`](Self::cmma_accumulator) through the manual-mma instruction.
    fn mma_accumulator<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] io: MmaIo,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`cmma_accumulator`](Self::cmma_accumulator) through the software instruction, run under
    /// `config`.
    fn block_accumulator<EA: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// [`block_accumulator`](Self::block_accumulator) for a reduction over `input`.
    fn block_reducer<EA: Numeric, In: Numeric>(
        &self,
        input: &Tile<In>,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA>;

    /// This accumulator opened with a per-plane shared-memory scratch for bouncing drains.
    fn with_scratch(self, #[comptime] scratch: Scratch) -> Self;

    /// Drain this accumulator into `dest`, cast to its element. `self` and `dest` must be
    /// indexed by the same region.
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
        // A block outliving the leaf cannot spread lines, so its lines and the sink's agree or fold.
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

/// The plane-resident grid an accumulator contracts in, in `form`, uninitialized.
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
