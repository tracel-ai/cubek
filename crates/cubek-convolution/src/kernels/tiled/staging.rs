use cubek_microkernel::matmul::MatmulOperand;

/// Which operands a launch reads through a stage, as against straight from global memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvStaging {
    /// The inputs are read by a stage fill and the output is written by the units' own cells,
    /// masked only where its extent needs it.
    Staged,
    /// As [`Staged`](Self::Staged), the output always masked: a cmma fragment's store writes its
    /// whole window and cannot be cut short, so the fragment is spilled through scratch and its
    /// cells written on under the mask.
    Bounced,
    /// Every operand is read or written from global memory, at the widths its accesses admit.
    Direct,
}

impl ConvStaging {
    /// Whether the output is masked whatever its extent.
    pub(super) fn masks_out(self) -> bool {
        self == Self::Bounced
    }

    pub(super) fn operands(self) -> &'static [MatmulOperand] {
        match self {
            Self::Staged | Self::Bounced => &[MatmulOperand::Lhs, MatmulOperand::Rhs],
            Self::Direct => &MatmulOperand::ALL,
        }
    }
}
