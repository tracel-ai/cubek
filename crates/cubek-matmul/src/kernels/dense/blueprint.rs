use crate::microkernel::MatmulPartitioning;
use crate::microkernel::{MatmulAxes, MatmulMicrokernel, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;

use crate::kernels::MatmulRoutineProblem;
use crate::stored::Packing;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_std::matmul_axes::{K, N};

/// What a routine's launch compiles: the microkernel, the partitioning that places its
/// work items, the width each operand binds at, and how the routine lays out shared memory.
///
/// One type for every routine, named per routine by an alias
/// ([`UnitRegistersMatmulRoutineBlueprint`](crate::kernels::UnitRegistersMatmulRoutineBlueprint),
/// [`PlaneShuffleMatmulRoutineBlueprint`](crate::kernels::PlaneShuffleMatmulRoutineBlueprint) and their
/// siblings): what differs between them is the microkernel, and the shared memory only the
/// routine that walks at that level decides. The order each operand binds in is the
/// microkernel's ([`MatmulMicrokernel::orders`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulBlueprint<M, SM = ()> {
    pub microkernel: M,
    pub partitioning: MatmulPartitioning,
    pub widths: MatmulOperands<usize>,
    /// How the routine lays out the shared memory its walk holds; `()` for one that holds none.
    pub shared_memory: SM,
}

impl<M: MatmulMicrokernel, SM> MatmulBlueprint<M, SM> {
    /// `microkernel` placed by `partitioning`, shared memory laid out as `shared_memory`: the
    /// blueprint once its widths are settled.
    ///
    /// To the requirements the routine and its microkernel enqueued in `vectorization`, the
    /// binding adds its own: each operand's buffer alignment in the order the microkernel binds the
    /// rhs in, and the masking of an operand whose innermost bound axis the tiles overhang. Each
    /// operand then binds at the widest width this device reads its element in that they admit.
    ///
    /// # Panics
    ///
    /// A partitioning holding its rows other than the microkernel does
    /// ([`MatmulMicrokernel::ROWS_KIND`]).
    pub fn new(
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        microkernel: M,
        partitioning: MatmulPartitioning,
        mut vectorization: Vectorization<MatmulOperand>,
        shared_memory: SM,
    ) -> Self {
        assert_eq!(
            partitioning.rows_kind(),
            M::ROWS_KIND,
            "a partitioning holds its rows as its microkernel does"
        );
        let orders = microkernel.orders();
        let alignments = orders.map(|operand, order| problem.alignment(operand, order));
        for operand in MatmulOperand::ALL {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: alignments.get(operand),
            });
            // A view starting partway into its buffer lines up only where its start does.
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    problem.elements.get(operand).size(),
                ),
            });
            if partitioning.overhangs(MatmulAxes::MNK.of(orders.innermost(operand))) {
                vectorization.require(VectorRequirement::Masked { operand });
            }
        }
        let widths = alignments.map(|operand, _| {
            vectorization.width(operand, |operand| {
                hw.vector_ceiling(problem.elements.get(operand))
            })
        });
        Self {
            microkernel,
            partitioning,
            widths,
            shared_memory,
        }
    }
}

impl<M> MatmulBlueprint<M> {
    /// `microkernel` placed by `partitioning` for a product against a weight packed as `rhs`
    /// says, each operand bound at the width the packing reads it at: a vector count of a float
    /// element means nothing for words. In values: the weight's a whole read, every axis it spans,
    /// the lhs and the output that read's run along the axis they share with it, one value across
    /// it. A packed lhs is contracted in those runs too, whole words of it
    /// ([`MatmulView::check_steps`](crate::kernels::MatmulView::check_steps)).
    pub fn packed(rhs: &Packing, microkernel: M, partitioning: MatmulPartitioning) -> Self {
        let operand = &rhs.operand;
        let lined = operand.lined();
        let along = |axis| match lined == axis {
            true => operand.read(axis),
            false => 1,
        };
        Self {
            microkernel,
            partitioning,
            widths: MatmulOperands {
                lhs: along(K),
                rhs: operand.read_values(),
                out: along(N),
            },
            shared_memory: (),
        }
    }
}
