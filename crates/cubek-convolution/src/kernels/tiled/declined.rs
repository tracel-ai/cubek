use cubecl::tile::launch::Refusal;

/// What a launch declines before preparing anything, whatever the device: operands, arguments or
/// extents that do not state one convolution the tiled kernels compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvDeclined {
    /// A bias, which is not part of the contraction.
    Bias,
    /// A weight or weight gradient whose channels are not the input's.
    Groups,
    /// Operands that are not 2D NHWC tensors and a `[co, kh, kw, ci]` weight or weight gradient,
    /// or arguments of another dimension.
    NotTwoDimensional,
    /// An operand that is quantized, whose scales the binding would drop.
    Quantized,
    /// The gradient of the input at a stride above 1, which reads the output gradient only at
    /// every stride-th position.
    DataBackwardStrided,
    /// An output of other channels than the weight's filters.
    OutChannels,
    /// Operands of different batches.
    Batch,
    /// A weight with no taps along a dimension.
    KernelEmpty,
    /// A stride or a dilation of zero, which no window steps by.
    WindowStepZero,
    /// An output that holds elements over a contraction of none: no input channels, or an empty
    /// batch or output gradient under a weight gradient.
    ContractionEmpty,
    /// A forward output whose spatial extent is not the one the input, the taps and the window
    /// yield.
    ForwardExtent,
    /// An output gradient whose extent, with the input, the weight gradient's taps and the
    /// window, does not state one convolution.
    WeightGradExtent,
    /// An input gradient whose extent, with the taps and the window, does not yield the output
    /// gradient's.
    DataGradExtent,
    /// A tiled tensor whose dims the view reorders, since the tiling labels the buffer's dims and
    /// would no longer match.
    PermutedTiled,
    /// A transposed forward, which the tiled convolution states as no operation of its own.
    Transposed,
}

impl ConvDeclined {
    const fn why(self) -> &'static str {
        match self {
            Self::Bias => "the tiled forward convolution adds no bias",
            Self::Groups => {
                "the tiled convolution contracts every input channel, so it has no groups"
            }
            Self::NotTwoDimensional => {
                "the tiled convolution reads 2D NHWC tensors and a [co, kh, kw, ci] weight"
            }
            Self::Quantized => {
                "the tiled convolution reads float operands, a quantized one would lose its scales"
            }
            Self::DataBackwardStrided => {
                "the tiled data backward convolution has a stride of 1, the gradient read at every stride-th position is not stated"
            }
            Self::OutChannels => "the tiled convolution's output channels are its weight's filters",
            Self::Batch => "the tiled convolution's operands share one batch",
            Self::KernelEmpty => {
                "the tiled convolution's weight has at least one tap along each spatial dimension"
            }
            Self::WindowStepZero => {
                "the tiled convolution's stride and dilation are at least 1 along each spatial dimension"
            }
            Self::ContractionEmpty => {
                "the tiled convolution contracts at least one cell for each element it writes, a sum of nothing is not stored"
            }
            Self::ForwardExtent => {
                "the forward output's extent is (input + 2 padding - dilation (taps - 1) - 1) / stride + 1 along each spatial dimension"
            }
            Self::WeightGradExtent => {
                "the output gradient's extent is (input + 2 padding - dilation (weight gradient taps - 1) - 1) / stride + 1 along each spatial dimension"
            }
            Self::DataGradExtent => {
                "the output gradient's extent is (input gradient + 2 padding - dilation (taps - 1) - 1) / stride + 1 along each spatial dimension"
            }
            Self::PermutedTiled => {
                "the tiled convolution reorders a tensor's dims, which it cannot do to a tiled one"
            }
            Self::Transposed => {
                "the tiled convolution has no transposed forward, which is its data backward at a stride"
            }
        }
    }
}

impl From<ConvDeclined> for Refusal {
    fn from(declined: ConvDeclined) -> Self {
        Refusal::Declined {
            why: declined.why(),
        }
    }
}
