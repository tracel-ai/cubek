//! A level-centric builder for a multi-level [`Space`]. Declare the axis extents once,
//! then one [`level`](LeveledTiling::level) per decomposition: its walk order, buffering, and
//! who works on which axes ([`LevelCuts`]). Each [`level`](LeveledTiling::level) maps 1:1 to the
//! [`Level`](super::Level) the [`Walk`](crate::Walk) consumes; no transpose.
//! [`Tiling::over`] is the same chain threading an [`OperandSet`] through each level
//! closure, so an operand states where it lives at the level that cuts it.

use crate::{Axis, ByAxis, Instruction, Space};

use super::{
    Buffering, ComputeScope, Distribution, Handout, OperandSet, Partitioner, Spatial, Spread,
    WalkOrder, Work,
};

/// How one axis is cut at one level: the sub-tile `edge` and how that level hands the tiles out.
///
/// What a level ends up holding, not what a caller writes: the two verbs on [`LevelCuts`] build
/// these, and nothing outside this module states one.
#[derive(Clone, Copy, Debug)]
struct Cut {
    edge: usize,
    dist: Distribution,
}

impl Cut {
    fn new(edge: usize, dist: Distribution) -> Self {
        Cut { edge, dist }
    }

    /// `edge`-sized tiles walked by every worker the level runs on.
    fn sequential(edge: usize) -> Self {
        Cut::new(edge, Distribution::Sequential)
    }
}

/// One decomposition level: its walk order, buffering, and the [`Cut`] for every axis.
struct LevelSpec {
    order: WalkOrder,
    buffering: Buffering,
    /// The axes this level distributes as one, if any ([`LevelCuts::distribute`]).
    work: Option<Work>,
    cuts: Vec<(Axis, Cut)>,
    /// Whether any operand stated a residence here ([`Tiling::over`] only). A level that cuts
    /// nothing but moves an operand is not null, so it is not droppable.
    moves_an_operand: bool,
}

impl LevelSpec {
    fn cut(&self, axis: Axis) -> Cut {
        self.cuts
            .iter()
            .find(|&&(a, _)| a == axis)
            .expect("checked in level()")
            .1
    }
}

/// The empty seed: declare [`extents`](Tiling::extents) to start adding levels.
pub struct Tiling;

impl Tiling {
    pub fn new() -> Self {
        Tiling
    }

    /// Declare every axis and its top extent, fixing the canonical axis order. Levels are
    /// added next; their cuts may come in any order and are realigned to this one.
    pub fn extents(self, extents: &[(Axis, usize)]) -> LeveledTiling {
        LeveledTiling {
            extents: extents.to_vec(),
            levels: Vec::new(),
            instruction: None,
        }
    }

    /// Build a space together with its operands, `extents` declaring every axis and its top
    /// extent as in [`extents`](Tiling::extents). Each level closure gets the cut collector and
    /// the operand set, so an operand's residence at a level is stated where the level is
    /// declared ([`Operand::stage`](crate::Operand::stage)); a level that states nothing for an
    /// operand leaves it in place. [`build`](OperandTiling::build) returns the space and the
    /// operands, sealed.
    pub fn over<'o, O: OperandSet>(
        operands: &'o mut O,
        extents: &[(Axis, usize)],
    ) -> OperandTiling<'o, O> {
        OperandTiling {
            tiling: Tiling::new().extents(extents),
            operands,
        }
    }
}

/// [`LeveledTiling`] threading an [`OperandSet`] through its level closures.
pub struct OperandTiling<'o, O> {
    tiling: LeveledTiling,
    operands: &'o mut O,
}

impl<O: OperandSet> OperandTiling<'_, O> {
    /// Add a decomposition level (coarse to fine): `f` states who works on which axes, on the
    /// collector, and states, per operand it materializes, where it lives here
    /// ([`Operand::stage`](crate::Operand::stage)).
    pub fn level(
        mut self,
        order: WalkOrder,
        buffering: Buffering,
        f: impl FnOnce(&mut LevelCuts, &mut O),
    ) -> Self {
        let mut cuts = LevelCuts::new();
        let index = self.tiling.levels.len();
        f(&mut cuts, self.operands);
        let moves_an_operand = self.operands.each().any(|o| o.stated_at(index));
        for operand in self.operands.each() {
            operand.close_level(index);
        }
        self.tiling
            .push(order, buffering, cuts.cuts, cuts.work, moves_an_operand);
        self
    }

    /// The last level, and what runs on the cells it cuts out. Walk order and buffering are
    /// fixed: one cell has nothing below it to double-buffer and no siblings to order. A space
    /// nothing contracts in ends with [`level`](Self::level) instead and states no instruction.
    pub fn instruction(
        mut self,
        instruction: Instruction,
        f: impl FnOnce(&mut LevelCuts, &mut O),
    ) -> Self {
        self.tiling = self.tiling.state_instruction(instruction);
        self.level(WalkOrder::RowMajor, Buffering::SINGLE, f)
    }

    /// Build the [`Space`]. The operands are the caller's own and stay theirs; this seals them,
    /// so one residence per level stands and every later [`stage`](crate::Operand::stage) panics.
    pub fn build(self) -> Space {
        // A dropped level is one no operand stated anything at, so its stage is the padded
        // `InPlace`; dropping it here keeps the residence column one entry per surviving level.
        let kept = self.tiling.kept_levels();
        for operand in self.operands.each() {
            operand.keep_levels(&kept);
            operand.seal();
        }
        self.tiling.build()
    }
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling::new()
    }
}

/// Builds a [`Space`] one level at a time: add levels with [`level`](LeveledTiling::level),
/// each configured by a closure that states, on a [`LevelCuts`], who works on which axes, then
/// end the chain with [`build`](LeveledTiling::build).
pub struct LeveledTiling {
    extents: Vec<(Axis, usize)>,
    levels: Vec<LevelSpec>,
    instruction: Option<Instruction>,
}

impl LeveledTiling {
    /// Add a decomposition level (coarse to fine) with its walk order and buffering; `cuts`
    /// states who works on which axes, on the [`LevelCuts`]. Where each operand *lives* at this
    /// level is stated by the operand, not here ([`Residence`](crate::Residence)).
    pub fn level(
        mut self,
        order: WalkOrder,
        buffering: Buffering,
        cuts: impl for<'a> FnOnce(&'a mut LevelCuts) -> &'a mut LevelCuts,
    ) -> Self {
        let mut level = LevelCuts::new();
        cuts(&mut level);
        self.push(order, buffering, level.cuts, level.work, false);
        self
    }

    /// Close `cuts` into a level. They must cover exactly the declared axes (any order);
    /// [`build`](Self::build) realigns them to the extents' canonical order.
    fn push(
        &mut self,
        order: WalkOrder,
        buffering: Buffering,
        cuts: Vec<(Axis, Cut)>,
        work: Option<Work>,
        moves_an_operand: bool,
    ) {
        // Per axis first: it names the one that is wrong, where the count only says the total
        // is off.
        for &(axis, _) in &self.extents {
            let stated = cuts.iter().filter(|&&(a, _)| a == axis).count();
            assert!(stated > 0, "LeveledTiling::level: axis {axis:?} has no cut");
            assert!(
                stated == 1,
                "LeveledTiling::level: axis {axis:?} is cut {stated} times; a level states each \
                 of its axes once, by `walk` or `distribute`"
            );
        }
        assert_eq!(
            cuts.len(),
            self.extents.len(),
            "LeveledTiling::level: {} cuts but {} axes declared",
            cuts.len(),
            self.extents.len()
        );
        self.levels.push(LevelSpec {
            order,
            buffering,
            work,
            cuts,
            moves_an_operand,
        });
    }

    /// The last level, and what runs on the cells it cuts out. See
    /// [`OperandTiling::instruction`].
    pub fn instruction(
        self,
        instruction: Instruction,
        cuts: impl for<'a> FnOnce(&'a mut LevelCuts) -> &'a mut LevelCuts,
    ) -> Self {
        self.state_instruction(instruction)
            .level(WalkOrder::RowMajor, Buffering::SINGLE, cuts)
    }

    /// Record what runs at the last level, without adding one.
    fn state_instruction(mut self, instruction: Instruction) -> Self {
        assert!(
            self.instruction.is_none(),
            "Tiling::instruction: stated once, already {:?}",
            self.instruction
        );
        self.instruction = Some(instruction);
        self
    }

    /// Which levels [`build`](Self::build) keeps, one flag per declared level.
    ///
    /// A level whose edges are the extents handed to it cuts nothing: [`Space::count`] is 1 on
    /// every axis. Drop it: the level is part of the [`Space`], and the [`Space`] is the
    /// kernel-cache key, so keeping it compiles the same program twice. Four things a level
    /// says that a cut does not, each keeping it: a deeper pipeline than its parent, work it
    /// distributes as one, an operand moving here, the instruction it carries, and being the only
    /// level left. That last one is
    /// not a fallback: a partitioned space separates a tile from the cells it is walked in, and
    /// a space with no level at all *is* its cell, which is a different space entirely.
    fn kept_levels(&self) -> Vec<bool> {
        // The extents handed to the next level: the top extents, then each kept level's edges.
        // Nothing is staged above the first level, so its parent buffers once.
        let mut parent = self.extents.clone();
        let mut parent_buffering = Buffering::SINGLE;
        let mut kept_any = false;
        let last = self.levels.len().saturating_sub(1);

        self.levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                let edges = self.edges(level);
                let last_standing = index == last && !kept_any;
                let keep = edges != parent
                    || level.buffering != parent_buffering
                    || level.work.is_some()
                    || level.moves_an_operand
                    || (self.instruction.is_some() && index == last)
                    || last_standing;
                if keep {
                    parent = edges;
                    parent_buffering = level.buffering;
                    kept_any = true;
                }
                keep
            })
            .collect()
    }

    /// One level's sub-tile edges, realigned to the canonical (extents) axis order, which is
    /// what makes them comparable to the extents handed down.
    fn edges(&self, level: &LevelSpec) -> Vec<(Axis, usize)> {
        self.extents
            .iter()
            .map(|&(a, _)| (a, level.cut(a).edge))
            .collect()
    }

    /// Build the [`Space`]: the extents, the stack of levels that cut something, and what runs
    /// once they are exhausted. Where each operand *lives* is the operands' own statement.
    pub fn build(self) -> Space {
        let kept = self.kept_levels();
        let mut space = Space::new(&self.extents);
        for (level, _) in self.levels.iter().zip(&kept).filter(|&(_, &keep)| keep) {
            let edges = self.edges(level);
            let dists: Vec<_> = self
                .extents
                .iter()
                .map(|&(a, _)| (a, level.cut(a).dist))
                .collect();
            let builder = match level.order {
                WalkOrder::RowMajor => {
                    Partitioner::row_major(ByAxis::new(&edges), ByAxis::new(&dists))
                }
                WalkOrder::Reversed => {
                    Partitioner::reversed(ByAxis::new(&edges), ByAxis::new(&dists))
                }
            };
            space =
                space.with_partitioner(builder.distributing(level.buffering, level.work.clone()));
        }
        match self.instruction {
            Some(instruction) => space.with_instruction(instruction),
            None => space,
        }
    }
}

/// Collects what one level says, in two verbs: [`distribute`](Self::distribute) hands some axes'
/// regions to a hardware scope's workers, [`walk`](Self::walk) leaves the rest to every one of
/// them. Between them they name each of the space's axes exactly once.
///
/// `&mut` receivers, so in a [`Tiling::over`] closure the two read as peer lines beside the
/// operands' [`stage`](crate::Operand::stage) statements.
pub struct LevelCuts {
    cuts: Vec<(Axis, Cut)>,
    work: Option<Work>,
}

impl LevelCuts {
    fn new() -> Self {
        LevelCuts {
            cuts: Vec::new(),
            work: None,
        }
    }

    /// Nobody owns these axes: every worker on this level covers all of their tiles, walking them
    /// one at a time. `axes` pairs each axis with its tile edge.
    ///
    /// The contraction of a matmul is the everyday one. So is any axis a level leaves alone,
    /// which is most of them at most levels.
    pub fn walk(&mut self, axes: &[(Axis, usize)]) -> &mut Self {
        self.cuts.extend(
            axes.iter()
                .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
        );
        self
    }

    /// Hand these axes' regions to `dist`'s workers, in order: as many workers as regions means
    /// one each, fewer means each takes a run. `axes` pairs each axis with its tile edge, and
    /// `dist` says who runs the tiles and how many of them there are ([`cubes`](crate::cubes),
    /// [`planes`](crate::planes), [`lanes`](crate::lanes) and their knobs).
    ///
    /// One axis, or one region each, is a box of the grid, so every axis gets a dial of its own
    /// and two lines make a two-dimensional box. Several axes sharing a stated count are read as a
    /// single index instead, so a worker takes a share of the whole rather than a box of it: a
    /// dial per axis always yields a box, and a share that begins inside one region and ends
    /// inside another is not one.
    ///
    /// That index runs over these axes' tiles at this level and one region of the level below, so
    /// a share can end part way through a region's own work. Where that region is an output tile
    /// and the level below walks the contraction, several workers end up holding pieces of the
    /// same cell, and the destination folds them exactly as it does under a dial
    /// ([`SplitShare`](crate::SplitShare)).
    ///
    /// An axis named here is not named by [`walk`](Self::walk): a level states each of its axes
    /// once.
    pub fn distribute(&mut self, dist: Spatial, axes: &[(Axis, usize)]) -> &mut Self {
        match dist.handout(axes.len()) {
            Handout::Dial => self.cuts.extend(
                axes.iter()
                    .map(|&(axis, edge)| (axis, Cut::new(edge, dist.into()))),
            ),
            Handout::OneIndex => {
                assert!(
                    self.work.is_none(),
                    "LevelCuts::distribute: this level already distributes work; state it once"
                );
                // A share is a range of one index, and lanes that reduce in registers have to
                // reach their reduction together. Ranges put them on different regions, so they
                // never would.
                assert!(
                    dist.scope() != ComputeScope::Unit,
                    "LevelCuts::distribute: the plane's lanes combine in registers, which needs \
                     them in lockstep, and lanes holding different shares never are. Distribute \
                     one axis at a time across them instead."
                );
                // A share is walked as a nest: consecutive steps of one region under one
                // accumulator, then the next region. Turns would put a different region under it
                // at every step.
                assert!(
                    dist.spread() == Spread::Contiguous,
                    "LevelCuts::distribute: a share is a run of the index, so its steps are \
                     consecutive; instances taking turns would leave no region long enough to \
                     accumulate in. Distribute one interleaved axis instead."
                );
                self.cuts.extend(
                    axes.iter()
                        .map(|&(axis, edge)| (axis, Cut::sequential(edge))),
                );
                self.work = Some(Work::new(
                    axes.iter().map(|&(axis, _)| axis).collect(),
                    dist,
                ));
            }
        }
        self
    }
}
