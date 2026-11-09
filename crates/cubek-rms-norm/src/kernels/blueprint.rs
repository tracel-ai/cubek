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
/// them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormBlueprint<M> {
    pub microkernel: M,
    pub partitioning: RmsNormPartitioning,
    pub width: usize,
    pub stage: CubeStage,
}

impl<M> RmsNormBlueprint<M> {
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
        partitioning: RmsNormPartitioning,
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
            if let Some(mask) = problem.store.mask(operand, partitioning.as_tile()) {
                vectorization.require(mask);
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

#[cfg(test)]
mod tests {
    use crate::kernels::{
        CubeGroupsRmsNormRoutine, CubeRmsNormRoutine, PlaneRmsNormRoutine, RmsNormRoutineProblem,
    };
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::rms_norm::RmsNormPartitioning;
    use cubek_microkernel::row_store::WIDTH;
    use cubek_std::devices;
    use cubek_std::routine::Routine;

    /// No setting of any routine overhangs a plain row, whatever its width: a run is whole lines
    /// of the row and divides it. What keeps
    /// [`RowStore::mask`](cubek_microkernel::row_store::RowStore::mask)'s plain arm unreached, so
    /// an output no wide masked line binds — a row stored in a product's tiles — never meets one;
    /// a setting that overhangs fails here before a launch does.
    #[test]
    fn no_setting_overhangs_a_plain_row() {
        let element = ElemType::Float(FloatKind::F16);
        let overhangs =
            |partitioning: &RmsNormPartitioning| partitioning.as_tile().overhangs(WIDTH);
        // Each routine prepares somewhere, so the check is never vacuous.
        let mut prepared = [0usize; 3];
        for hw in [
            devices::strix_halo(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for (rows, dim) in [(1, 96), (1, 1003), (7, 2560), (3, 4097), (1, 11008)] {
                let problem = RmsNormRoutineProblem::dense(rows, dim, element);
                for settings in CubeRmsNormRoutine.settings(&hw) {
                    if let Ok((blueprint, _)) = CubeRmsNormRoutine.prepare(&hw, &problem, &settings)
                    {
                        assert!(
                            !overhangs(&blueprint.partitioning),
                            "cube {dim} {settings:?}"
                        );
                        prepared[0] += 1;
                    }
                }
                for settings in CubeGroupsRmsNormRoutine.settings(&hw) {
                    if let Ok((blueprint, _)) =
                        CubeGroupsRmsNormRoutine.prepare(&hw, &problem, &settings)
                    {
                        assert!(
                            !overhangs(&blueprint.partitioning),
                            "groups {dim} {settings:?}"
                        );
                        prepared[1] += 1;
                    }
                }
                for settings in PlaneRmsNormRoutine.settings(&hw) {
                    if let Ok((blueprint, _)) =
                        PlaneRmsNormRoutine.prepare(&hw, &problem, &settings)
                    {
                        assert!(
                            !overhangs(&blueprint.partitioning),
                            "plane {dim} {settings:?}"
                        );
                        prepared[2] += 1;
                    }
                }
            }
        }
        assert!(prepared.iter().all(|&count| count > 0), "{prepared:?}");
    }
}
