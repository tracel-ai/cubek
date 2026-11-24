use crate::kernels::select::problem::OperandKind;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::Product;
use crate::kernels::tune::key::{MatmulTuneKey, create_key};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulOperands;
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
        let operands = self.operands();
        Product::of([operands.lhs, operands.rhs])
    }

    /// What each operand is: its inputs as they are stored, its output float.
    pub(crate) fn operands(&self) -> MatmulOperands<OperandKind> {
        MatmulOperands {
            lhs: self.lhs.tensor.kind(),
            rhs: self.rhs.tensor.kind(),
            out: OperandKind::Float,
        }
    }

    /// This call writing into its destination, allocated here at the product's shape where the
    /// caller gave none: a float product in its inputs' element, a packed one in the call's. What an
    /// autotune measures on, so every candidate and the winner write the one buffer the call
    /// returns.
    pub(crate) fn with_destination(mut self) -> Self {
        if self.destination.is_some() {
            return self;
        }
        let elem = match (&self.lhs.tensor, &self.rhs.tensor) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(_)) => lhs.elem,
            _ => self.out,
        };
        let (handle, _) = PlainTensor::empty(&self.client, self.product_shape().into(), elem);
        self.destination = Some(handle);
        self
    }

    /// The product's shape: the inputs' broadcast batch dimensions, then the lhs's rows and the
    /// rhs's columns, read off their logical shapes.
    fn product_shape(&self) -> Vec<usize> {
        let logical = |input: &MatmulCallInput| {
            input
                .meta
                .logical_shape()
                .expect("a tensor's tiling describes its own rank")
        };
        let (lhs, rhs) = (logical(&self.lhs), logical(&self.rhs));
        let rank = lhs.rank();
        (0..rank - 2)
            .map(|dim| lhs[dim].max(rhs[dim]))
            .chain([lhs[rank - 2], rhs[rank - 1]])
            .collect()
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
pub type FrameworkDequantize = Arc<dyn Fn() -> Result<PlainTensor, Decline> + Send + Sync>;

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
