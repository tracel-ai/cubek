use crate::rms_norm::RmsNormOperand;
use crate::row_store::RowStore;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;

/// A row of this kind as its microkernels see it: the element every operand is stored in, the
/// dim the row spans, which every owner holds whole, and how the row is stored, which decides the
/// runs an owner may split it into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormWorkItem {
    pub element: ElemType,
    /// The normalized dim: what the mean divides by.
    pub dim: usize,
    /// How the rows are stored, which decides the runs an owner may write.
    pub store: RowStore,
}

impl RmsNormWorkItem {
    /// The width every operand binds at on `hw`: what the operands' one requirement, read along
    /// the whole row ([`RmsNormOperand::require_along_run`]), settles at the device's widest read.
    /// The operands are bound as dense rows starting on a whole read, so nothing else narrows it;
    /// a microkernel sizes its registers at it, and a routine declines a blueprint that binds
    /// another.
    pub fn width(&self, hw: &Hardware) -> usize {
        let mut vectorization = Vectorization::new();
        RmsNormOperand::require_along_run(self.dim, &mut vectorization);
        vectorization.width(RmsNormOperand::Input, |_| hw.vector_ceiling(self.element))
    }

    /// Lines of the row at [`width`](Self::width).
    pub fn lines(&self, hw: &Hardware) -> usize {
        self.dim.div_ceil(self.width(hw)).max(1)
    }
}
