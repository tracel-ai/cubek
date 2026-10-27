use crate::attention_backward::AttentionBackwardOperand;
use crate::matmul::{MatmulOperand, MatmulWorkItem};
use cubecl::tile::launch::Vectorization;
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

impl<M: Microkernel<Operand = MatmulOperand, Partitioning = ()>> BoundProduct<M> {
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
/// its own product: their registers added, on the instruction they share.
pub(crate) fn estimate_beside<M>(
    products: impl IntoIterator<Item = (M, MatmulWorkItem)>,
    planes: CubePlanes,
    estimate: &mut KernelEstimate,
) where
    M: Microkernel<WorkItem = MatmulWorkItem, Partitioning = ()>,
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
