use crate::attention::{AttentionOperand, AttentionOperands, AttentionWorkItem};
use cubecl::tile::Space;
use cubecl::tile::launch::Delivery;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::Microkernel;

/// A microkernel of this kind, as its routines and their blueprint read it beside what every
/// microkernel states.
pub trait AttentionMicrokernel:
    Microkernel<WorkItem = AttentionWorkItem, Operand = AttentionOperand, CubeSpace = Space>
    + core::fmt::Display
{
    /// The order its score reads its rhs, the keys, in, which the launch binds them in:
    /// [`InBufferOrder`](BindingOrder::InBufferOrder), `[s, d]` as they lie, or
    /// [`Stated`](BindingOrder::Stated), `[d, s]` along the keys, a stride over the same
    /// buffer.
    fn key_order(&self) -> BindingOrder;

    /// `item` as its fragments read it: its operands at the element they reach the instruction
    /// in. As they are stored, unless its instruction reads them in another element.
    fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        *item
    }

    /// How the keys and values it binds at `widths` are moved into its stages, which the launch
    /// states on both: [`SyncPerUnit`](Delivery::SyncPerUnit) for one that stages nothing.
    fn delivery(&self, _widths: &AttentionOperands<usize>, _item: &AttentionWorkItem) -> Delivery {
        Delivery::SyncPerUnit
    }
}
