use crate::kernels::tensor::{PlainTensor, StoredTensor};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use std::sync::Arc;

/// One product a plan runs: its two inputs, each as it is stored, the element it is written in,
/// and the device it runs on as the plan prepares for it.
#[derive(Clone)]
pub struct MatmulCall {
    pub hw: Hardware,
    pub client: Client,
    pub lhs: MatmulCallInput,
    pub rhs: MatmulCallInput,
    /// The element the product is written in.
    pub out: ElemType,
}

/// One input of a [`MatmulCall`]: the tensor as it is stored, and, where it is packed, how the
/// caller's framework dequantizes it.
///
/// The framework's dequantize reads every scheme, and the kernels here do not: a plan reaches for
/// it where no kernel reads the scheme, and where the decode declines an input. It is the caller's
/// own tensor that the framework reads, so the caller binds it here and the plan never sees it.
#[derive(Clone)]
pub struct MatmulCallInput {
    pub tensor: StoredTensor,
    pub framework_dequantize: Option<FrameworkDequantize>,
}

/// The framework's dequantize of one packed input, into the element the product is written in.
pub type FrameworkDequantize = Arc<dyn Fn() -> Result<PlainTensor, Refusal> + Send + Sync>;

impl MatmulCallInput {
    /// A float input, which no dequantize is asked about.
    pub fn plain(tensor: PlainTensor) -> Self {
        Self {
            tensor: StoredTensor::Plain(tensor),
            framework_dequantize: None,
        }
    }

    /// The input the product reads in place of this one: the float tensor `handle` holds.
    pub(crate) fn replaced_by(handle: TensorHandle) -> Self {
        Self::plain(PlainTensor::of_handle(handle))
    }
}
