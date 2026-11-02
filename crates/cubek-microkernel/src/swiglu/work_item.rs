use crate::row_store::RowStore;
use crate::swiglu::SwigluOperand;
use cubecl::ir::ElemType;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;

/// A row of this kind as its microkernels see it: the element every operand is stored in, the
/// width of each half of the fused row, which the output takes, and how the output is stored,
/// which decides the runs a cube may encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwigluWorkItem {
    pub element: ElemType,
    /// The width of the gate, of the up and of the output: half the fused row.
    pub dim: usize,
    /// How the output is stored, which decides the runs a cube may encode.
    pub store: RowStore,
}

impl SwigluWorkItem {
    /// The width every operand binds at on `hw`: what the operands' one requirement, read along
    /// the whole row ([`SwigluOperand::require_along_run`]), settles at the device's widest read.
    /// The halves are bound as runs of `dim` starting on a whole read, so nothing else narrows it;
    /// a microkernel sizes its run at it, and a routine declines a blueprint that binds another.
    pub fn width(&self, hw: &Hardware) -> usize {
        let mut vectorization = Vectorization::new();
        SwigluOperand::require_along_run(self.dim, &mut vectorization);
        vectorization.width(SwigluOperand::Input, |_| hw.vector_ceiling(self.element))
    }

    /// Lines of the row at [`width`](Self::width).
    pub fn lines(&self, hw: &Hardware) -> usize {
        self.dim.div_ceil(self.width(hw)).max(1)
    }
}
