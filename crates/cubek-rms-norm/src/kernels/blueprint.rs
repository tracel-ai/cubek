//! An RMS-norm launch's compile-time decisions.

use crate::kernels::RmsNormRoutineProblem;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::rms_norm::{RmsNormOperand, RmsNormPartitioning};
use cubek_microkernel::row_store::CubeStage;
use cubek_std::Hardware;

/// What a routine declines with where a blueprint binds its operands at a width other than the
/// one its microkernel sized its registers at: the operands were not bound as dense rows.
pub const RMS_NORM_WIDTH_UNPLANNED: Refusal = Refusal::Declined {
    why: "the RMS norm binds its operands at the width its microkernel is sized at, a dense row's",
};

/// What a launch of this kind compiles: the microkernel, the partitioning that places its rows,
/// the one width every operand binds at — each is read a line at a time beside the others — and
/// the stage a cube writes its rows into before they reach a packed output.
///
/// One type for every routine of this kind, named per routine by an alias: what differs between
/// them is the microkernel `M` and the way `P` its boxes are handed out
/// ([`Partition`](cubek_std::routine::Partition)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormBlueprint<M, P> {
    pub microkernel: M,
    pub partitioning: RmsNormPartitioning<P>,
    pub width: usize,
    pub stage: CubeStage,
}

impl<M, P> RmsNormBlueprint<M, P> {
    /// `microkernel` placed by `partitioning`: the blueprint once its width is settled.
    ///
    /// To the requirements the routine and its microkernel enqueued in `vectorization`, the
    /// binding adds its own: each operand's buffer alignment, where its view starts, and the
    /// masking of every operand where the tiles overhang the row. The operands then bind at the
    /// widest width this device reads their element in that they all admit.
    pub fn new(
        hw: &Hardware,
        problem: &RmsNormRoutineProblem,
        microkernel: M,
        partitioning: RmsNormPartitioning<P>,
        mut vectorization: Vectorization<RmsNormOperand>,
    ) -> Self {
        for operand in RmsNormOperand::ALL {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: problem.elem_alignments.get(operand),
            });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    problem.element.size(),
                ),
            });
            if problem.store.overhangs_width(partitioning.as_tile()) {
                vectorization.require(VectorRequirement::masked(
                    operand,
                    partitioning.as_tile().space(),
                    problem.store.contiguous_axis(),
                ));
            }
        }
        let width = vectorization.width(RmsNormOperand::Input, |_| {
            hw.vector_ceiling(problem.element)
        });
        let stage = problem.store.cube_stage(&partitioning.cube_box());
        Self {
            microkernel,
            partitioning,
            width,
            stage,
        }
    }

    /// This blueprint, where it binds its operands at `width`, the one its microkernel sized its
    /// registers at.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_WIDTH_UNPLANNED`].
    pub fn planned_at(self, width: usize) -> Result<Self, Refusal> {
        match self.width == width {
            true => Ok(self),
            false => Err(RMS_NORM_WIDTH_UNPLANNED),
        }
    }
}
