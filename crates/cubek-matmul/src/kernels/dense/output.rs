use crate::kernels::MatmulRoutineProblem;
use crate::kernels::MatmulTensors;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Geometry;
use cubek_microkernel::matmul::MatmulDestination;

/// An output buffer, in the operands' element: the tensor the launch hands back where it
/// allocated it, and the view of it the launch binds, its rows merged where the launch reads every
/// row whole.
#[derive(Debug, Clone)]
pub struct MatmulOutputBuffer {
    pub(super) tensor: Option<TensorHandle>,
    pub(super) bound: PlainTensor,
}

/// Where a product goes, as a launch that stores its boxes through the TMA engine asks of it: the
/// tensor a buffer the launch allocates is bound as, which a descriptor addresses, and nothing
/// for a sink a caller binds, which has no address.
pub trait MatmulOutput {
    /// Where the product lands, as the problem states it.
    const DESTINATION: MatmulDestination;

    /// The tensor the launch binds the product into, where it has one.
    fn addressed(&self) -> Option<&PlainTensor>;

    /// The product of `lhs` and `rhs` landing here.
    fn problem(&self, lhs: &PlainTensor, rhs: &PlainTensor) -> MatmulRoutineProblem;

    /// The element the product is written in, beside the activation `lhs`.
    fn element(&self, lhs: &StoredTensor) -> ElemType;
}

impl MatmulOutput for MatmulOutputBuffer {
    const DESTINATION: MatmulDestination = MatmulDestination::Buffer;

    fn addressed(&self) -> Option<&PlainTensor> {
        Some(&self.bound)
    }

    fn problem(&self, lhs: &PlainTensor, rhs: &PlainTensor) -> MatmulRoutineProblem {
        MatmulRoutineProblem::new(lhs, rhs, &self.bound)
    }

    fn element(&self, _lhs: &StoredTensor) -> ElemType {
        self.bound.elem
    }
}

impl MatmulOutput for Geometry {
    const DESTINATION: MatmulDestination = MatmulDestination::Sink;

    fn addressed(&self) -> Option<&PlainTensor> {
        None
    }

    fn problem(&self, lhs: &PlainTensor, rhs: &PlainTensor) -> MatmulRoutineProblem {
        MatmulRoutineProblem::sink(lhs, rhs, self)
    }

    fn element(&self, lhs: &StoredTensor) -> ElemType {
        MatmulTensors::packed_output(lhs)
    }
}
