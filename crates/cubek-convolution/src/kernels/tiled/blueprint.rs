use super::{ConvPartitioning, ConvProblem};
use cubecl::tile::Scratch;
use cubecl::tile::launch::{MaskedExtent, Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::matmul::plane::{PlaneFragmentDrain, PlaneFragmentMicrokernel};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::routine::{CubeResources, Grid};

/// What a launch compiles: the microkernel, the partitioning that places it, and the width each
/// operand binds at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvBlueprint<M> {
    pub microkernel: M,
    pub partitioning: ConvPartitioning,
    pub widths: MatmulOperands<usize>,
    /// Whether the output is masked whatever its extent: a cmma fragment stores whole through its
    /// intrinsic into an unmasked output, which gave wrong values; a masked one bounces through a
    /// window.
    pub masks_out: bool,
}

impl<M> ConvBlueprint<M> {
    /// `microkernel`, whose planes are `plane`, placed by `partitioning`, and the grid that spawns
    /// it.
    ///
    /// To the requirements the routine enqueued in `vectorization`, the binding adds its own: each
    /// input's buffer alignment and where its view starts, and the masking of an innermost run
    /// the tiles overhang. Each input then binds at the widest width this device reads its element in
    /// that they admit, and the output one element wide, as a manual fragment writes it.
    ///
    /// # Errors
    ///
    /// A cube or a grid the device does not spawn, or one holding more shared memory than it gives.
    pub fn prepared(
        hw: &Hardware,
        problem: &ConvProblem,
        (microkernel, plane): (M, &PlaneFragmentMicrokernel),
        partitioning: ConvPartitioning,
        mut vectorization: Vectorization<MatmulOperand>,
    ) -> Result<(Self, Grid), Refusal> {
        for operand in [MatmulOperand::Lhs, MatmulOperand::Rhs] {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: problem.alignments.get(operand),
            });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    problem.elements.get(operand).size(),
                ),
            });
            let innermost = problem.operation.roles().inner.get(operand);
            if partitioning.overhangs(innermost) {
                // A padded input's lines cannot be tested a line at a time at an edge a tile
                // overhangs: it is read one element wide there.
                vectorization.require(
                    match (operand, problem.window.underflows(problem.operation)) {
                        (MatmulOperand::Lhs, true) => VectorRequirement::Masked {
                            operand,
                            extent: MaskedExtent::Unknown,
                        },
                        _ => VectorRequirement::masked(operand, partitioning.space(), innermost),
                    },
                );
            }
        }
        let width = |operand| {
            let ceiling = hw.vector_ceiling(problem.elements.get(operand));
            vectorization.width(operand, |_| ceiling)
        };
        let widths = MatmulOperands {
            lhs: width(MatmulOperand::Lhs),
            rhs: width(MatmulOperand::Rhs),
            out: 1,
        };
        let bounces = plane.instruction.drain() == PlaneFragmentDrain::Bounce;
        let windows = match bounces {
            true => {
                partitioning.planes() * plane.scratch_bytes(&problem.work_item(), Scratch::OneTile)
            }
            false => 0,
        };
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: partitioning.stage_bytes(&problem.elements).max(windows),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let blueprint = Self {
            microkernel,
            partitioning,
            widths,
            masks_out: bounces,
        };
        Ok((blueprint, grid))
    }
}
