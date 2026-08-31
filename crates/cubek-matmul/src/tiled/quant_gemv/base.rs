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
/// [`select`](QuantGemvRoutine::select).
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
        // Widen the fold across blocks only as far as the blocks deal out evenly; the lanes
        // that remain carry rows.
        let block_lanes = (1..=free)
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
