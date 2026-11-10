use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of the causal conv.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalConvOperand {
    /// `[batch, channel]`: the newest value of every channel over a window, or `[batch,
    /// stream_position, channel]` over a prompt.
    Input,
    /// `[channel, tap]`: the filter of every channel, its newest tap last.
    Filters,
    /// `[channel]`: the bias of every channel, where the conv has one.
    Bias,
    /// `[store_slot, ring_column, channel]`: the past values of every channel, a ring a slot.
    Ring,
    /// `[batch]`: the positions each row has convolved already.
    Lengths,
    /// `[batch, channel]`, or `[batch, position, channel]` over a prompt: the convolved channels.
    Output,
}

impl CausalConvOperand {
    /// The operands a line of channels runs along.
    pub const CHANNELS: [Self; 3] = [Self::Input, Self::Ring, Self::Output];

    /// What every microkernel of the conv requires: the operands along the channels share one
    /// width that divides `channels`, and the filters, the bias and the lengths are read an element
    /// at a time.
    pub fn require(channels: usize, vectorization: &mut Vectorization<Self>) {
        for operand in Self::CHANNELS {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: channels,
            });
            vectorization.require(VectorRequirement::Together {
                operand: Self::Input,
                with: operand,
            });
        }
        for operand in [Self::Filters, Self::Bias, Self::Lengths] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalConvOperandMap<T> {
    pub input: T,
    pub filters: T,
    pub bias: T,
    pub ring: T,
    pub lengths: T,
    pub output: T,
}

impl<T: Copy> CausalConvOperandMap<T> {
    /// The same value for every operand.
    pub fn splat(value: T) -> Self {
        Self {
            input: value,
            filters: value,
            bias: value,
            ring: value,
            lengths: value,
            output: value,
        }
    }

    /// The value `operand` holds.
    pub fn get(&self, operand: CausalConvOperand) -> T {
        match operand {
            CausalConvOperand::Input => self.input,
            CausalConvOperand::Filters => self.filters,
            CausalConvOperand::Bias => self.bias,
            CausalConvOperand::Ring => self.ring,
            CausalConvOperand::Lengths => self.lengths,
            CausalConvOperand::Output => self.output,
        }
    }
}
