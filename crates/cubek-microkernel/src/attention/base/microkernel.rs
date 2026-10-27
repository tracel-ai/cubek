use crate::attention::{AttentionOperand, AttentionPartitioning, AttentionWorkItem};
use crate::matmul::MatmulOrder;
use cubek_std::microkernel::Microkernel;

/// A microkernel of this kind, as its routines and their blueprint read it beside what every
/// microkernel states.
pub trait AttentionMicrokernel:
    Microkernel<
        WorkItem = AttentionWorkItem,
        Operand = AttentionOperand,
        Partitioning = AttentionPartitioning,
    > + core::fmt::Display
{
    /// The order its score reads its rhs, the keys, in, which the launch binds them in:
    /// [`InBufferOrder`](MatmulOrder::InBufferOrder), `[s, d]` as they lie, or
    /// [`Stated`](MatmulOrder::Stated), `[d, s]` along the keys, a stride over the same
    /// buffer.
    fn key_order(&self) -> MatmulOrder;

    /// `item` as its fragments read it: its operands at the element they reach the instruction
    /// in. As they are stored, unless its instruction reads them in another element.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        *item
    }
}
