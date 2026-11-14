use crate::attention::plane::ScoreHolder;
use crate::attention_backward::AttentionBackwardOperand;
use crate::matmul::{MatmulOperand, MatmulWorkItem};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{FragmentSource, Instruction, ShapedInstruction};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// One of a backward microkernel's multiplications as its pass binds it: the shared microkernel
/// that runs it, and the operand its lhs, its rhs and its product are, `None` for one the walk
/// keeps in registers or a window instead of binding.
pub(crate) struct BoundProduct<M> {
    pub microkernel: M,
    pub lhs: Option<AttentionBackwardOperand>,
    pub rhs: Option<AttentionBackwardOperand>,
    pub out: Option<AttentionBackwardOperand>,
}

impl<M: Microkernel<Operand = MatmulOperand, CubeSpace = ()>> BoundProduct<M> {
    /// The shared microkernel's requirements, restated on the operands this product binds.
    pub fn restate(self, vectorization: &mut Vectorization<AttentionBackwardOperand>) {
        let mut product = Vectorization::new();
        self.microkernel.vectorize(&(), &mut product);
        vectorization.restate(product, |operand| match operand {
            MatmulOperand::Lhs => self.lhs,
            MatmulOperand::Rhs => self.rhs,
            MatmulOperand::Out => self.out,
        });
    }
}

/// The work of multiplications a unit holds at once, each shared microkernel setting the work of
/// its own product: their registers added.
pub(crate) fn estimate_beside<M>(
    products: impl IntoIterator<Item = (M, MatmulWorkItem)>,
    planes: CubePlanes,
    estimate: &mut KernelEstimate,
) where
    M: Microkernel<WorkItem = MatmulWorkItem, CubeSpace = ()>,
{
    let mut work = None;
    for (microkernel, product) in products {
        microkernel.estimate(&product, &(), planes, estimate);
        let block = estimate.work.take();
        work = match work {
            None => block,
            Some(held) => block.map(|block| held.beside(block)),
        };
    }
    estimate.work = work;
}

/// The tile's pick for one of a backward plane's sums kept between its contractions, `item` over
/// the plane's `tile`, kept as `holder` says: drained into the plane's window, the sum of its
/// operands' values; held in registers, a sum contracted again
/// ([`Instruction::for_stage_accumulator`]), which the kernel opens against its lhs.
pub(crate) fn score_pick(
    hw: &Hardware,
    holder: ScoreHolder,
    item: &MatmulWorkItem,
    tile: [usize; 3],
) -> Option<ShapedInstruction> {
    let (lhs, rhs) = (
        FragmentSource::Values(item.elements.lhs),
        FragmentSource::Values(item.elements.rhs),
    );
    let (device, target) = (&hw.properties, &hw.target);
    match holder {
        ScoreHolder::Window => Instruction::for_accumulator(
            device,
            target,
            ComputeScope::Plane,
            lhs,
            rhs,
            item.sum,
            tile,
        ),
        ScoreHolder::Fragments => {
            Instruction::for_stage_accumulator(device, target, lhs, lhs, item.sum, tile)
        }
    }
}

/// The tile's pick for a product that reads one of those sums as its `A`, `item` over the plane's
/// `tile`: from the plane's window, cast to the operands' element, where `holder` drains it; held
/// in registers, runs of `held`'s cells as they lie.
pub(crate) fn reading_pick(
    hw: &Hardware,
    holder: ScoreHolder,
    held: ShapedInstruction,
    item: &MatmulWorkItem,
    tile: [usize; 3],
) -> Option<ShapedInstruction> {
    let lhs = match holder {
        ScoreHolder::Window => FragmentSource::Values(item.elements.lhs),
        ScoreHolder::Fragments => FragmentSource::Fragments {
            instruction: held.instruction,
            cell: held.cell(),
        },
    };
    let rhs = FragmentSource::Values(item.elements.rhs);
    Instruction::for_accumulator(
        &hw.properties,
        &hw.target,
        ComputeScope::Plane,
        lhs,
        rhs,
        item.sum,
        tile,
    )
}
