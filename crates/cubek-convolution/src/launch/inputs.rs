use cubecl::prelude::TensorBinding;
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
