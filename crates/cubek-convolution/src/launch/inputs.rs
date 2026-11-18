use crate::components::ConvolutionOperation;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Axis;
use cubecl::tile::layout::{PhysicalAxisMap, Scale};
use cubek_std::InputBinding;
use serde::{Deserialize, Serialize};

/// Spatial convolution arguments (stride / beginning padding / dilation per spatial dim).
///
/// End padding is represented by the spatial extent of the output binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "ArgsRepr", try_from = "ArgsRepr")]
pub struct ConvolutionArgs<const N_SPATIAL: usize> {
    pub stride: [usize; N_SPATIAL],
    /// Padding at the beginning of each spatial dimension.
    pub padding: [usize; N_SPATIAL],
    pub dilation: [usize; N_SPATIAL],
}

impl<const N_SPATIAL: usize> ConvolutionArgs<N_SPATIAL> {
    /// The window of each spatial dimension.
    pub fn windows(&self) -> [SpatialWindow; N_SPATIAL] {
        std::array::from_fn(|dim| SpatialWindow {
            stride: self.stride[dim],
            padding: self.padding[dim],
            dilation: self.dilation[dim],
        })
    }
}

/// The serialized form of [`ConvolutionArgs`], whose arrays serde has no impl for at any length.
#[derive(Debug, Serialize, Deserialize)]
struct ArgsRepr {
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
}

impl<const N_SPATIAL: usize> From<ConvolutionArgs<N_SPATIAL>> for ArgsRepr {
    fn from(args: ConvolutionArgs<N_SPATIAL>) -> Self {
        Self {
            stride: args.stride.to_vec(),
            padding: args.padding.to_vec(),
            dilation: args.dilation.to_vec(),
        }
    }
}

impl<const N_SPATIAL: usize> TryFrom<ArgsRepr> for ConvolutionArgs<N_SPATIAL> {
    type Error = String;

    fn try_from(repr: ArgsRepr) -> Result<Self, String> {
        let dims = |dims: Vec<usize>| {
            <[usize; N_SPATIAL]>::try_from(dims)
                .map_err(|dims| format!("{} spatial dims, not {N_SPATIAL}", dims.len()))
        };
        Ok(Self {
            stride: dims(repr.stride)?,
            padding: dims(repr.padding)?,
            dilation: dims(repr.dilation)?,
        })
    }
}

/// The stride, beginning padding and dilation of one spatial dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialWindow {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl SpatialWindow {
    /// The extent a window of `kernel` taps yields over an `input` of that extent, zero where the
    /// dilated taps do not fit the padded input.
    pub fn output(&self, input: usize, kernel: usize) -> usize {
        let padded = input + 2 * self.padding;
        let span = self.dilation * kernel.saturating_sub(1) + 1;
        match padded.checked_sub(span) {
            Some(room) => room / self.stride + 1,
            None => 0,
        }
    }

    /// The input position a forward window reads at output position `out` and tap `tap`,
    /// `out * stride + tap * dilation - padding`: negative before the input's start.
    pub fn position(&self, out: usize, tap: usize) -> isize {
        (out * self.stride + tap * self.dilation) as isize - self.padding as isize
    }

    /// [`Self::position`] as a map over the axes `out` and `tap`, for a tile projection.
    pub fn position_map(&self, out: Axis, tap: Axis) -> PhysicalAxisMap {
        PhysicalAxisMap::affine(&[(out, self.stride), (tap, self.dilation)])
            .shifted(-(self.padding as isize))
    }

    /// The gradient position the data backward reads at input position `pos` and tap `tap`,
    /// `pos + padding - tap * dilation`, at stride 1: negative before the gradient's start, as a
    /// map over the axes `pos` and `tap`.
    pub fn gradient_position_map(&self, pos: Axis, tap: Axis) -> PhysicalAxisMap {
        PhysicalAxisMap::scaled(&[
            (pos, Scale::Static(1)),
            (tap, Scale::Static(-(self.dilation as isize))),
        ])
        .shifted(self.padding as isize)
    }
}

#[allow(clippy::large_enum_variant)]
/// Per-operation tensor bindings supplied to `launch_ref`.
///
/// Each variant carries exactly the bindings the corresponding operation needs.
/// The discriminant maps 1:1 to `ConvolutionOperation`.
pub enum ConvolutionInputs {
    Forward {
        input: InputBinding,
        weight: InputBinding,
        bias: Option<InputBinding>,
        out: TensorBinding,
    },
    BackwardData {
        out_grad: InputBinding,
        weights: InputBinding,
        in_grad: TensorBinding,
    },
    BackwardWeight {
        input: InputBinding,
        out_grad: InputBinding,
        weight_grad: TensorBinding,
    },
}

impl ConvolutionInputs {
    pub fn operation(&self) -> ConvolutionOperation {
        match self {
            ConvolutionInputs::Forward { .. } => ConvolutionOperation::Forward,
            ConvolutionInputs::BackwardData { .. } => ConvolutionOperation::BackwardData,
            ConvolutionInputs::BackwardWeight { .. } => ConvolutionOperation::BackwardWeight,
        }
    }
}
