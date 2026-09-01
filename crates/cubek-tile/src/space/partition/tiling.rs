//! A level-centric builder for a multi-level [`Space`]. [`Tiling::over`] threads the operands
//! through, then one [`level`](OperandTiling::level) per decomposition: its walk order,
//! buffering, the per-axis [`Cut`], and where each operand it materializes lives. Each level maps
//! 1:1 to the [`Level`](super::Level) the [`Walk`](crate::Walk) consumes; no transpose.

use crate::{Axis, ByAxis, Instruction, Space};

use super::{Buffering, CubeAxis, Distribution, OperandSet, Partitioner, WalkOrder};

/// How one axis is cut at one level: the sub-tile `edge` and how that level hands the
/// tiles out. Constructors name the common distributions; [`Cut::new`] takes any.
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    pub edge: usize,
    pub dist: Distribution,
}

impl Cut {
    pub fn new(edge: usize, dist: Distribution) -> Self {
        Cut { edge, dist }
    }

    /// `edge`-sized tiles dealt one-per-cube along `axis`.
    pub fn cube(axis: CubeAxis, edge: usize) -> Self {
        Cut::new(edge, Distribution::cube(axis))
    }

    /// `edge`-sized tiles dealt one-per-plane (worker thread).
    pub fn plane(edge: usize) -> Self {
        Cut::new(edge, Distribution::plane())
    }

    /// `edge`-sized tiles spread across the plane's lanes. The lane count is the hardware
    /// `plane_size`, stamped at launch by [`Space::launcher`]; the cut carries only the
    /// intent (see [`Distribution::unit`]).
    pub fn unit(edge: usize) -> Self {
        Cut::new(edge, Distribution::unit())
    }

    /// `edge`-sized tiles walked sequentially by one instance.
    pub fn sequential(edge: usize) -> Self {
        Cut::new(edge, Distribution::Sequential)
    }
}

/// One decomposition level: its walk order, buffering, and the [`Cut`] for every axis.
struct LevelSpec {
    order: WalkOrder,
    buffering: Buffering,
    cuts: Vec<(Axis, Cut)>,
    /// Whether any operand stated a residence here. A level that cuts nothing but moves an
    /// operand is not null, so it is not droppable.
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

/// The builder's entry point, and the only one: a space is always built over an operand set,
/// even when that set is empty.
pub struct Tiling;

impl Tiling {
    /// Build a space together with its operands. `extents` declares every axis and its top
    /// extent, fixing the canonical axis order; cuts may then come in any order and are
    /// realigned to it.
    ///
    /// Each level closure gets the cut collector and the operand set, so an operand's residence
    /// at a level is stated where the level is declared
    /// ([`Operand::stage`](crate::Operand::stage)); a level that states nothing for an operand
    /// leaves it in place. [`build`](OperandTiling::build) returns the space and seals the
    /// operands.
    ///
    /// A space with no operand to place passes the empty set: `Tiling::over(&mut (), …)`.
    pub fn over<'o, O: OperandSet>(
        operands: &'o mut O,
        extents: &[(Axis, usize)],
    ) -> OperandTiling<'o, O> {
        OperandTiling {
            extents: extents.to_vec(),
            levels: Vec::new(),
            instruction: None,
            operands,
        }
    }
}

/// Builds a [`Space`] one level at a time, carrying the operands each level places. Add levels
/// with [`level`](Self::level), end the chain with [`instruction`](Self::instruction) where
/// something contracts at the last one, then [`build`](Self::build).
pub struct OperandTiling<'o, O> {
    extents: Vec<(Axis, usize)>,
    levels: Vec<LevelSpec>,
    instruction: Option<Instruction>,
    operands: &'o mut O,
}

impl<O: OperandSet> OperandTiling<'_, O> {
    /// Add a decomposition level (coarse to fine): `f` hangs the per-axis [`Cut`]s off the
    /// collector and states, per operand it materializes, where it lives here
    /// ([`Operand::stage`](crate::Operand::stage)).
    pub fn level(
        mut self,
        order: WalkOrder,
        buffering: Buffering,
        f: impl FnOnce(&mut LevelCuts, &mut O),
    ) -> Self {
        let mut cuts = LevelCuts { cuts: Vec::new() };
        let index = self.levels.len();
        f(&mut cuts, self.operands);
        let moves_an_operand = self.operands.each().any(|o| o.stated_at(index));
        for operand in self.operands.each() {
            operand.close_level(index);
        }
        self.push(order, buffering, cuts.cuts, moves_an_operand);
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
        assert!(
            self.instruction.is_none(),
            "Tiling::instruction: stated once, already {:?}",
            self.instruction
        );
        self.instruction = Some(instruction);
        self.level(WalkOrder::RowMajor, Buffering::SINGLE, f)
    }

    /// Build the [`Space`]: the extents, the stack of levels that cut something, and what runs
    /// once they are exhausted. The operands are the caller's own and stay theirs; this seals
    /// them, so one residence per level stands and every later
    /// [`stage`](crate::Operand::stage) panics.
    pub fn build(self) -> Space {
        // A dropped level is one no operand stated anything at, so its stage is the padded
        // `InPlace`; dropping it here keeps the residence column one entry per surviving level.
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
            space = space.with_partitioner(builder.buffered(level.buffering));
        }
        for operand in self.operands.each() {
            operand.keep_levels(&kept);
            operand.seal();
        }
        match self.instruction {
            Some(instruction) => space.with_instruction(instruction),
            None => space,
        }
    }

    /// Close `cuts` into a level. They must cover exactly the declared axes (any order);
    /// [`build`](Self::build) realigns them to the extents' canonical order.
    fn push(
        &mut self,
        order: WalkOrder,
        buffering: Buffering,
        cuts: Vec<(Axis, Cut)>,
        moves_an_operand: bool,
    ) {
        assert_eq!(
            cuts.len(),
            self.extents.len(),
            "Tiling::level: {} cuts but {} axes declared",
            cuts.len(),
            self.extents.len()
        );
        for &(axis, _) in &self.extents {
            assert!(
                cuts.iter().any(|&(a, _)| a == axis),
                "Tiling::level: axis {axis:?} has no cut"
            );
        }
        self.levels.push(LevelSpec {
            order,
            buffering,
            cuts,
            moves_an_operand,
        });
    }

    /// Which levels [`build`](Self::build) keeps, one flag per declared level.
    ///
    /// A level whose edges are the extents handed to it cuts nothing: [`Space::count`] is 1 on
    /// every axis. Drop it: the level is part of the [`Space`], and the [`Space`] is the
    /// kernel-cache key, so keeping it compiles the same program twice. Four things a level
    /// says that a cut does not, each keeping it: a deeper pipeline than its parent, an operand
    /// moving here, the instruction it carries, and being the only level left. That last one is
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
}

/// Collects one level's per-axis [`Cut`]s, via [`axis`](Self::axis) for a single axis and
/// [`axes`](Self::axes) to hand a whole group the same cut. `&mut` receivers, so the cuts and
/// the operands' [`stage`](crate::Operand::stage) statements read as peer lines.
pub struct LevelCuts {
    cuts: Vec<(Axis, Cut)>,
}

impl LevelCuts {
    /// One axis gets `cut`.
    pub fn axis(&mut self, axis: Axis, cut: Cut) -> &mut Self {
        self.cuts.push((axis, cut));
        self
    }

    /// Every axis in `axes` gets the same `cut` (e.g. all batch axes pinned alike).
    pub fn axes(&mut self, axes: &[Axis], cut: Cut) -> &mut Self {
        self.cuts.extend(axes.iter().map(|&a| (a, cut)));
        self
    }
}
