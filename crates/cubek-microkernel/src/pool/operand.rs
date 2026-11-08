use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of a pooling that are bound to a buffer. An average's weights are
/// computed, never read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolOperand {
    /// `[b, ih, iw, c]`, gathered over the window of each output position.
    Input,
    /// `[b, oh, ow, c]`.
    Output,
}

impl PoolOperand {
    /// What every microkernel of this kind requires of the operands, each unit pooling runs of
    /// `channels` channels: a line of the input is a line of the output, and it divides the run.
    pub fn require_along_channels(channels: usize, vectorization: &mut Vectorization<Self>) {
        for operand in [Self::Input, Self::Output] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: channels,
            });
        }
        vectorization.require(VectorRequirement::Together {
            operand: Self::Input,
            with: Self::Output,
        });
    }
}
