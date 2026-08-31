//! The QuantGemv routine: what a decode gemv launch is compiled to do, and how the shape and
//! the device settle it.
//!
//! # Supported
//!
//! - A weight packed into `u32` words along the contraction (`d_in`), its scales blocking the
//!   same axis: one scale per `(output row, block of K)`. That is the serving form a decode
//!   step streams — the buffer's contiguous direction is the walk.
//! - Any field [`QuantValue`] names. The device's reported vector cap is not among the
//!   questions: the weight binds one `u32` *word* per line whatever the packing factor expands
//!   to, and the activation's line is bytes, not lanes of a binding the cap governs.
//! - Activation, scale and output element types stated independently; the contraction runs in
//!   the element the words decode to.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - A scale block that does not cover whole words, or a `d_in` the blocks do not tile.
//! - A `d_out` no plan's row strip tiles.

use std::fmt::Display;

use cubecl::quant::scheme::QuantValue;

use crate::definition::MatmulSetupError;

/// The shape of a decode gemv, in the units the plan reasons about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantGemvProblem {
    /// The weight's output dimension: rows of the lhs, and of the result.
    pub d_out: usize,
    /// The contraction: the weight's packed direction.
    pub d_in: usize,
    /// Activation rows projected in one launch (1 at decode).
    pub rows: usize,
    /// The stored field, which names its own width and sign.
    pub field: QuantValue,
    /// Values one scale covers, along the contraction.
    pub block: usize,
}

impl QuantGemvProblem {
    /// Values packed into one stored word: the width a packed line is served at, and the run of
    /// the contraction one lane takes per step.
    pub fn factor(&self) -> usize {
        32 / self.field.size_bits()
    }

    /// Blocks along the contraction.
    pub fn blocks(&self) -> usize {
        self.d_in / self.block
    }
}

/// A fully-resolved plan: a strip of output rows per cube, a run of them per plane, a run per
/// aligned lane group, and the group's lanes interleaving the contraction between them.
///
/// The lanes that split `K` hold partials of the same cell and fold inside the plane, which is
/// what the two `Unit` cuts on `(KB, KI)` state. Their instance product with the row groups is
/// exactly the plane width — the engine's geometry contract, satisfied by construction in
/// [`QuantGemvRoutine::blueprint`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantGemvBlueprint {
    /// Output rows one cube covers.
    pub rows_per_cube: usize,
    /// Output rows one plane covers.
    pub rows_per_plane: usize,
    /// Output rows one lane owns — the register instruction's `mr`. Past one, a lane issues
    /// that many weight loads a step at addresses a whole row apart: they cannot fuse into a
    /// wider read, and that is the point — the bytes are identical, the loads are in flight
    /// together.
    pub rows_per_lane: usize,
    /// Lanes splitting the position *inside* a block: each takes one stored word.
    pub inside_lanes: usize,
    /// Lanes splitting the blocks themselves: each takes one block per step. A cut cuts `KB` or
    /// cuts `KI` and cannot straddle two axes, so a group wider than one block is spelled here.
    pub block_lanes: usize,
}

impl QuantGemvBlueprint {
    /// Lanes that share one output row and fold their partials together.
    pub fn group_lanes(&self) -> usize {
        self.inside_lanes * self.block_lanes
    }

    /// Aligned lane groups in a plane, each carrying its own rows.
    pub fn groups(&self) -> usize {
        self.rows_per_plane / self.rows_per_lane
    }

    /// Reject a plan whose cuts do not tile, or whose lanes do not close a plane.
    #[allow(clippy::result_large_err)]
    pub fn validate(
        &self,
        problem: &QuantGemvProblem,
        plane_dim: usize,
    ) -> Result<(), MatmulSetupError> {
        let refuse = |what: String| Err(MatmulSetupError::InvalidConfig(Box::new(what)));
        if self.rows_per_lane == 0 || !self.rows_per_plane.is_multiple_of(self.rows_per_lane) {
            return refuse(format!(
                "QuantGemv: {} rows a plane do not divide into {}-row lanes",
                self.rows_per_plane, self.rows_per_lane
            ));
        }
        if self.groups() * self.group_lanes() != plane_dim {
            return refuse(format!(
                "QuantGemv: {} row groups of {} lanes do not close a {plane_dim}-lane plane",
                self.groups(),
                self.group_lanes()
            ));
        }
        if !problem.d_out.is_multiple_of(self.rows_per_cube)
            || !self.rows_per_cube.is_multiple_of(self.rows_per_plane)
        {
            return refuse(format!(
                "QuantGemv: {} output rows do not tile into {}-row cubes of {}-row planes",
                problem.d_out, self.rows_per_cube, self.rows_per_plane
            ));
        }
        if !problem.blocks().is_multiple_of(self.block_lanes) {
            return refuse(format!(
                "QuantGemv: {} blocks of K do not deal out to {} lanes",
                problem.blocks(),
                self.block_lanes
            ));
        }
        if self.inside_lanes * problem.factor() != problem.block {
            return refuse(format!(
                "QuantGemv: {} lanes of {}-value words do not cover a {}-value block",
                self.inside_lanes,
                problem.factor(),
                problem.block
            ));
        }
        Ok(())
    }
}

/// The knobs a caller may lean on. None yet: the plan is a closed function of the shape and the
/// plane width, and a pin is [`Forced`](crate::routine::BlueprintStrategy::Forced).
#[derive(Clone, Debug, Default)]
pub struct QuantGemvStrategy;

impl Display for QuantGemvStrategy {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

/// Pairs the [`QuantGemvStrategy`] knobs with the [`QuantGemvBlueprint`] plan.
pub struct QuantGemvRoutine;

impl crate::routine::Routine<()> for QuantGemvRoutine {
    type Strategy = QuantGemvStrategy;
    type Blueprint = QuantGemvBlueprint;
}

/// Output rows one lane may own, widest first. A preference the search narrows rather than a
/// constant: a `d_out` the four-row strip misses is served two rows wide instead of dropping to
/// one row a plane. Four leads because it wins or ties everywhere the shipped gemv was swept,
/// and eight collapses once the `K` walk is long — its block staying live across three times as
/// many steps against the same registers.
const ROW_BLOCKS: [usize; 3] = [4, 2, 1];

/// Planes per cube, widest first.
const PLANE_STRIPS: [usize; 3] = [4, 2, 1];

/// Lanes that share one output row and fold their partials together.
///
/// A target rather than a constant: how many of them fit *inside* a block is the block's
/// business, so a group reaching this width is spelled as lanes of `KI` and lanes of `KB`
/// together. Eight is where the fold was measured to sit — past it the butterfly costs more
/// steps than the extra memory-level parallelism buys, and below it a lane's run of the
/// contraction gets long enough that the row block goes cold.
const GROUP_LANES: usize = 8;

impl QuantGemvRoutine {
    /// Resolve `strategy` into a validated plan for `problem` on a `plane_dim`-lane plane.
    #[allow(clippy::result_large_err)]
    pub fn blueprint(
        strategy: &crate::routine::BlueprintStrategy<(), QuantGemvRoutine>,
        problem: &QuantGemvProblem,
        plane_dim: usize,
    ) -> Result<QuantGemvBlueprint, MatmulSetupError> {
        Self::validate_problem(problem, plane_dim)?;
        let blueprint = match strategy {
            crate::routine::BlueprintStrategy::Forced(blueprint) => *blueprint,
            crate::routine::BlueprintStrategy::Inferred(_) => Self::select(problem, plane_dim)?,
        };
        blueprint.validate(problem, plane_dim)?;
        Ok(blueprint)
    }

    /// What no plan can rescue: a word the device cannot serve as a line, or blocks that do not
    /// tile the contraction in whole words.
    #[allow(clippy::result_large_err)]
    fn validate_problem(
        problem: &QuantGemvProblem,
        plane_dim: usize,
    ) -> Result<(), MatmulSetupError> {
        let refuse = |what: String| Err(MatmulSetupError::InvalidConfig(Box::new(what)));
        let factor = problem.factor();
        if !problem.block.is_multiple_of(factor) {
            return refuse(format!(
                "QuantGemv: a {}-value block does not cover whole {factor}-value words",
                problem.block
            ));
        }
        if !problem.d_in.is_multiple_of(problem.block) {
            return refuse(format!(
                "QuantGemv: {}-value blocks do not tile a {}-deep contraction",
                problem.block, problem.d_in
            ));
        }
        let inside_lanes = problem.block / factor;
        if !plane_dim.is_multiple_of(inside_lanes) {
            return refuse(format!(
                "QuantGemv: a block takes {inside_lanes} lanes, which do not divide a {plane_dim}-lane plane"
            ));
        }
        Ok(())
    }

    /// The plan: a block of `K` is covered by `block / factor` lanes, whatever is left of the
    /// plane carries rows, and the row strip is the widest the output tiles.
    #[allow(clippy::result_large_err)]
    fn select(
        problem: &QuantGemvProblem,
        plane_dim: usize,
    ) -> Result<QuantGemvBlueprint, MatmulSetupError> {
        // One block of `K` takes this many lanes; the rest of the plane is free to carry rows
        // or to take further blocks.
        let inside_lanes = (problem.block / problem.factor()).min(plane_dim);
        let free = plane_dim / inside_lanes;
        // Reach [`GROUP_LANES`] by taking whole blocks as well, as far as the blocks deal out
        // evenly. Every lane past that carries rows instead: a wider fold is a longer drain
        // against no more bytes in flight.
        let wanted = GROUP_LANES.div_ceil(inside_lanes);
        let block_lanes = (1..=free.min(wanted))
            .filter(|lanes| free.is_multiple_of(*lanes) && problem.blocks().is_multiple_of(*lanes))
            .max()
            .unwrap_or(1);
        let groups = free / block_lanes;

        // Widest row block first, then the widest strip of planes under it.
        let (rows_per_lane, planes) = ROW_BLOCKS
            .iter()
            .copied()
            .flat_map(|rows_per_lane| {
                PLANE_STRIPS
                    .iter()
                    .copied()
                    .map(move |planes| (rows_per_lane, planes))
            })
            .find(|&(rows_per_lane, planes)| {
                problem
                    .d_out
                    .is_multiple_of(planes * groups * rows_per_lane)
            })
            .ok_or_else(|| {
                MatmulSetupError::InvalidConfig(Box::new(format!(
                    "QuantGemv: {} output rows are not tiled by any strip of {groups} lane groups",
                    problem.d_out
                )))
            })?;

        Ok(QuantGemvBlueprint {
            rows_per_cube: planes * groups * rows_per_lane,
            rows_per_plane: groups * rows_per_lane,
            rows_per_lane,
            inside_lanes,
            block_lanes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::BlueprintStrategy;

    /// The qwen3-8b decode projections, `(d_in, d_out)`.
    const QWEN3_8B: [(usize, usize); 5] = [
        (4096, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
        (4096, 151936),
    ];

    fn q4(d_in: usize, d_out: usize) -> QuantGemvProblem {
        QuantGemvProblem {
            d_out,
            d_in,
            rows: 1,
            field: QuantValue::Q4S,
            block: 32,
        }
    }

    fn plan(problem: &QuantGemvProblem, plane_dim: usize) -> QuantGemvBlueprint {
        QuantGemvRoutine::blueprint(&BlueprintStrategy::default(), problem, plane_dim)
            .unwrap_or_else(|e| panic!("no plan for {problem:?} on {plane_dim} lanes: {e}"))
    }

    /// A group is eight lanes whatever the block costs to cover, and the rest of the plane
    /// carries rows. q4's 32-value block takes four lanes of `KI`, so the group reaches eight
    /// by taking two blocks; q8's word is half as wide, so eight lanes of `KI` cover the block
    /// alone and nothing splits `KB`.
    #[test]
    fn a_lane_group_is_eight_lanes_however_the_block_divides() {
        let q4 = plan(&q4(4096, 4096), 32);
        assert_eq!((q4.inside_lanes, q4.block_lanes), (4, 2));
        assert_eq!(q4.group_lanes(), 8);
        assert_eq!(q4.groups(), 4);

        let q8 = plan(
            &QuantGemvProblem {
                field: QuantValue::Q8S,
                ..q4_problem()
            },
            32,
        );
        assert_eq!((q8.inside_lanes, q8.block_lanes), (8, 1));
        assert_eq!(q8.group_lanes(), 8);
    }

    fn q4_problem() -> QuantGemvProblem {
        q4(4096, 4096)
    }

    /// Every qwen3-8b decode projection gets the widest strip: four rows a lane, four groups a
    /// plane, four planes a cube. `lm_head`'s 151936 columns are not a power of two, so it is
    /// the one that says the search is a search.
    #[test]
    fn every_qwen_projection_takes_a_plan() {
        for (d_in, d_out) in QWEN3_8B {
            let blueprint = plan(&q4(d_in, d_out), 32);
            assert_eq!(
                blueprint.rows_per_lane, 4,
                "{d_in}x{d_out} narrowed its row block"
            );
            assert_eq!(blueprint.rows_per_cube, 64);
        }
    }

    /// A plane too narrow for the block's own lanes still plans: the lanes that cover a block
    /// are all of it, and the rows go one to a group.
    #[test]
    fn a_narrow_plane_spends_itself_on_the_block() {
        let blueprint = plan(&q4(4096, 4096), 4);
        assert_eq!((blueprint.inside_lanes, blueprint.block_lanes), (4, 1));
        assert_eq!(blueprint.groups(), 1);
    }

    /// What no plan rescues, said as a refusal rather than a wrong answer.
    #[test]
    fn a_block_that_splits_a_word_is_refused() {
        let problem = QuantGemvProblem {
            block: 4, // four values, where a q4 word holds eight
            ..q4_problem()
        };
        let error = QuantGemvRoutine::blueprint(&BlueprintStrategy::default(), &problem, 32)
            .expect_err("a block inside a word has no spelling");
        assert!(
            format!("{error:?}").contains("whole"),
            "the refusal must name the word it splits, got {error:?}"
        );
    }
}
