use super::{ConvPartitioning, ConvProblem, ConvStaging};
use cubecl::tile::launch::{MaskedExtent, Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::routine::{CubeResources, Grid};

/// What a launch compiles: the microkernel, the partitioning that places it, and the width each
/// operand binds at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConvBlueprint<M> {
    pub microkernel: M,
    pub partitioning: ConvPartitioning,
    pub widths: MatmulOperands<usize>,
    /// Whether the output is masked whatever its extent ([`ConvStaging::Bounced`]).
    pub masks_out: bool,
}

impl<M> TileConvBlueprint<M> {
    /// `microkernel` placed by `partitioning`, and the grid that spawns it with the resources
    /// `cube` asks of each cube. Each operand `staging` names binds at the widest width this
    /// device reads its element in that `vectorization` admits.
    ///
    /// # Errors
    ///
    /// A cube or a grid the device does not spawn, or one holding more shared memory than it gives.
    pub fn new(
        hw: &Hardware,
        problem: &ConvProblem,
        microkernel: M,
        partitioning: ConvPartitioning,
        vectorization: Vectorization<MatmulOperand>,
        staging: ConvStaging,
        cube: CubeResources,
    ) -> Result<(Self, Grid), Refusal> {
        let widths = Self::widths(
            hw,
            problem,
            &partitioning,
            vectorization,
            staging.operands(),
        );
        let grid = Grid::new(hw, partitioning.tile_partitioning(), cube)?;
        let blueprint = Self {
            microkernel,
            partitioning,
            widths,
            masks_out: staging.masks_out(),
        };
        Ok((blueprint, grid))
    }

    /// To the requirements the routine enqueued in `vectorization`, the binding adds its own for
    /// each of `operands`: its buffer's alignment and where its view starts, and the masking of an
    /// innermost run the tiles overhang. Each then binds at the widest width this device reads its
    /// element in that they admit; an operand not listed binds one element wide.
    fn widths(
        hw: &Hardware,
        problem: &ConvProblem,
        partitioning: &ConvPartitioning,
        mut vectorization: Vectorization<MatmulOperand>,
        operands: &[MatmulOperand],
    ) -> MatmulOperands<usize> {
        for &operand in operands {
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
            let innermost = problem.roles().inner.get(operand);
            if partitioning.overhangs(innermost) {
                // Where a read can fall before the gathered operand's start (padding, or the
                // data backward's taps), its vectors cannot be tested a vector at a time at an edge
                // a tile overhangs: it is read one element wide there.
                vectorization.require(match (operand, problem.underflows()) {
                    (MatmulOperand::Lhs, true) => VectorRequirement::Masked {
                        operand,
                        extent: MaskedExtent::Unknown,
                    },
                    (MatmulOperand::Lhs, false) | (MatmulOperand::Rhs | MatmulOperand::Out, _) => {
                        VectorRequirement::masked(operand, partitioning.space(), innermost)
                    }
                });
            }
        }
        let width = |operand| match operands.contains(&operand) {
            true => {
                let ceiling = hw.vector_ceiling(problem.elements.get(operand));
                vectorization.width(operand, |_| ceiling)
            }
            false => 1,
        };
        MatmulOperands {
            lhs: width(MatmulOperand::Lhs),
            rhs: width(MatmulOperand::Rhs),
            out: width(MatmulOperand::Out),
        }
    }
}
