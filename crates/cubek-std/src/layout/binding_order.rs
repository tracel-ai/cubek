use cubecl::tile::launch::{Arg, Labelled};

/// The order an operand is bound in: the axes the problem names, or the order its buffer steps.
///
/// A col rhs (an `[n, k]` buffer behind a `[k, n]` view) bound over the axes the product names
/// steps its innermost bound axis by `k`, so it is read one element at a time; bound in its
/// buffer's order, the same bytes are read in lines, as attention's keys are against their head
/// dim. Only a fragment that loads its operand col-major reads one so, and the two orders compile
/// two kernels, so the microkernel states which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingOrder {
    /// As the problem names its axes: `[k, n]` for a matmul's rhs.
    Stated,
    /// As its buffer steps them: `[n, k]` for a col rhs.
    InBufferOrder,
}

impl core::fmt::Display for BindingOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stated => write!(f, "stated"),
            Self::InBufferOrder => write!(f, "in-buffer-order"),
        }
    }
}

impl BindingOrder {
    /// `operand` bound in this order: as the problem names it, or in the order its buffer steps.
    /// The one place a binding reads the order; which axis it leaves innermost is each problem's
    /// to say, from the operand it binds.
    pub fn applied(self, operand: Arg<'_, Labelled>) -> Arg<'_, Labelled> {
        match self {
            Self::Stated => operand,
            Self::InBufferOrder => operand.in_stride_order(),
        }
    }
}
