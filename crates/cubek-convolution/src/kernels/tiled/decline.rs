use cubecl::tile::launch::Decline;

/// Why the tiled convolution declines a launch: operands, arguments or extents that do not state
/// one convolution it computes, or a setting the device cannot serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvDecline {
    /// A bias, which is not part of the contraction.
    Bias,
    /// A weight or weight gradient whose channels are not the input's.
    Groups,
    /// A convolution over no spatial dim.
    NoSpatialDim,
    /// A convolution over more spatial dims than the axis labels number.
    TooManySpatialDims,
    /// Operands whose rank is not two more than the arguments' spatial dims: channels last
    /// activations and a `[co, taps.., ci]` weight or weight gradient.
    RankMismatch,
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
    /// A device that offers no instruction of the setting's kind for the problem's elements.
    InstructionUnoffered,
    /// A device whose instruction of the setting's kind the tile does not pick for the problem
    /// at the rows the setting names.
    RowsNotPicked,
    /// A setting that asks for a warpgroup MMA, whose stages only the TMA engine lands and which
    /// this convolution's fill never does.
    WgmmaUnserved,
}

impl ConvDecline {
    const fn why(self) -> &'static str {
        match self {
            Self::Bias => "the tiled forward convolution adds no bias",
            Self::Groups => {
                "the tiled convolution contracts every input channel, so it has no groups"
            }
            Self::NoSpatialDim => "the tiled convolution has at least one spatial dimension",
            Self::TooManySpatialDims => {
                "the tiled convolution labels each axis in a byte, which bounds how many spatial dimensions it has"
            }
            Self::RankMismatch => {
                "the tiled convolution reads channels last tensors and a [co, taps.., ci] weight, one dim per spatial dimension of its arguments"
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
            Self::InstructionUnoffered => {
                "the device offers no instruction of this kind for the problem's elements"
            }
            Self::RowsNotPicked => {
                "the tile picks an instruction of other rows than the setting's for this problem"
            }
            Self::WgmmaUnserved => "a warpgroup MMA reads its stages as the TMA engine lands them",
        }
    }
}

impl From<ConvDecline> for Decline {
    fn from(decline: ConvDecline) -> Self {
        Decline::Unserved { why: decline.why() }
    }
}
