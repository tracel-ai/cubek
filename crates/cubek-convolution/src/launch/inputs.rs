use cubecl::{Runtime, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::components::ConvolutionOperation;

/// Spatial convolution arguments (stride / padding / dilation per spatial dim).
#[derive(Clone, Debug)]
pub struct ConvolutionArgs<const N_SPATIAL: usize> {
    pub stride: [usize; N_SPATIAL],
    pub padding: [usize; N_SPATIAL],
    pub dilation: [usize; N_SPATIAL],
}

#[allow(clippy::large_enum_variant)]
/// Per-operation tensor bindings supplied to `launch_ref`.
///
/// Each variant carries exactly the bindings the corresponding operation needs.
/// The discriminant maps 1:1 to `ConvolutionOperation`.
pub enum ConvolutionInputs<R: Runtime> {
    Forward {
        input: InputBinding<R>,
        weight: InputBinding<R>,
        bias: Option<InputBinding<R>>,
        out: TensorBinding<R>,
    },
    BackwardData {
        out_grad: InputBinding<R>,
        weights: InputBinding<R>,
        in_grad: TensorBinding<R>,
    },
    BackwardWeight {
        input: InputBinding<R>,
        out_grad: InputBinding<R>,
        weight_grad: TensorBinding<R>,
    },
}

impl<R: Runtime> ConvolutionInputs<R> {
    pub fn operation(&self) -> ConvolutionOperation {
        match self {
            ConvolutionInputs::Forward { .. } => ConvolutionOperation::Forward,
            ConvolutionInputs::BackwardData { .. } => ConvolutionOperation::BackwardData,
            ConvolutionInputs::BackwardWeight { .. } => ConvolutionOperation::BackwardWeight,
        }
    }
}
