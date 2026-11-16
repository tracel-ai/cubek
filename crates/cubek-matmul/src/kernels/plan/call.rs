use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::Product;
use crate::kernels::tune::key::{MatmulTuneKey, create_key};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::metadata::Metadata;
use cubek_std::Hardware;
use cubek_std::plan::BoxedPlan;
use std::sync::Arc;

/// A plan of the matmul family: what a product's launch tries, on one [`MatmulCall`].
pub type MatmulPlan = BoxedPlan<MatmulCall, TensorHandle>;

/// One product a plan runs: its two inputs, each as it is stored, the element it is written in,
/// and the device it runs on as the plan prepares for it.
#[derive(Clone)]
pub struct MatmulCall {
    pub hw: Hardware,
    pub client: Client,
    /// The device's identity, which an autotune keys its measurements by.
    pub tune_id: CubeTuneId,
    pub lhs: MatmulCallInput,
    pub rhs: MatmulCallInput,
    /// The element the product is written in.
    pub out: ElemType,
    /// The buffer the product is written into, where the caller allocated it at the product's
    /// shape; a launch allocates one otherwise.
    pub destination: Option<TensorHandle>,
}

impl MatmulCall {
    /// The key an autotune measures the product under: its extents, the element it is written in,
    /// and each input as it lies.
    pub fn key(&self) -> MatmulTuneKey {
        create_key(
            self.out,
            (&self.lhs.meta, self.lhs.tensor.kind()),
            (&self.rhs.meta, self.rhs.tensor.kind()),
        )
    }

    /// The kind of product its inputs make: what a candidate listed for the other kind declines.
    pub(crate) fn product(&self) -> Product {
        Product::of([self.lhs.tensor.kind(), self.rhs.tensor.kind()])
    }

    /// What a tuning round measures for `key`: the call's rhs, and a zero float lhs as many rows
    /// as the key's bucket stands for, or the call's lhs where none can be laid out alike.
    pub fn representative(key: &MatmulTuneKey, call: &Self) -> Self {
        let grown = match &call.lhs.tensor {
            // The rows lie along the lhs's second-to-last axis.
            StoredTensor::Plain(lhs) => (lhs.binding.shape.len().checked_sub(2))
                .and_then(|rows| lhs.zeros_grown(&call.client, rows, key.m)),
            StoredTensor::Packed(_) => None,
        };
        // A grown lhs makes a product of another shape than the caller's buffer, so the round's
        // launches allocate their own.
        match grown {
            Some(lhs) => Self {
                lhs: MatmulCallInput::plain(lhs),
                destination: None,
                ..call.clone()
            },
            None => call.clone(),
        }
    }
}

/// One input of a [`MatmulCall`]: the tensor as it is stored, how it lies as its caller states it,
/// and, where it is packed, how the caller's framework dequantizes it.
///
/// The framework's dequantize reads every scheme, and the kernels here do not: a plan reaches for
/// it where no kernel reads the scheme, and where the decode declines an input. It is the caller's
/// own tensor that the framework reads, so the caller binds it here and the plan never sees it.
#[derive(Clone)]
pub struct MatmulCallInput {
    pub tensor: StoredTensor,
    /// Its logical shape and strides, which a tuning key reads.
    pub meta: Metadata,
    pub framework_dequantize: Option<FrameworkDequantize>,
}

/// The framework's dequantize of one packed input, into the element the product is written in.
pub type FrameworkDequantize = Arc<dyn Fn() -> Result<PlainTensor, Refusal> + Send + Sync>;

impl MatmulCallInput {
    /// A float input, which no dequantize is asked about.
    pub fn plain(tensor: PlainTensor) -> Self {
        Self {
            meta: tensor.meta(),
            tensor: StoredTensor::Plain(tensor),
            framework_dequantize: None,
        }
    }
}
