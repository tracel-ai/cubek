use crate::kernels::indexed::routed::RoutedMatmulTensors;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::{MatmulTuneKey, create_key};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
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
    /// The device's identity, which a table keys its measurements by.
    pub tune_id: CubeTuneId,
    pub lhs: MatmulCallInput,
    pub rhs: MatmulCallInput,
    /// The element the product is written in.
    pub out: ElemType,
    /// Where each lhs row reads the slabs of a stacked rhs its routes name: the routed product
    /// they state, read as stored once for every candidate of a table.
    pub routes: Option<RoutedMatmulTensors>,
    /// The buffer the product is written into, where the caller allocated it at the product's
    /// shape; a launch allocates one otherwise.
    pub destination: Option<TensorHandle>,
}

impl MatmulCall {
    /// The key a table measures the product under: its extents, the element it is written in,
    /// and each input as it lies.
    pub fn key(&self) -> MatmulTuneKey {
        if let Some(routes) = &self.routes {
            return routes.key();
        }
        create_key(
            self.out,
            (&self.lhs.meta, self.lhs.tensor.kind()),
            (&self.rhs.meta, self.rhs.tensor.kind()),
        )
    }

    /// What a tuning round measures for `key`: the call's rhs, and a zero float lhs as many rows
    /// as the key's bucket stands for, or the call's lhs where none can be laid out alike. A routed
    /// product is measured on the call's own operands: its rows ride the batches, and its routes
    /// are what the call's are.
    pub fn representative(key: &MatmulTuneKey, call: &Self) -> Self {
        let grown = match (&call.routes, &call.lhs.tensor) {
            (None, StoredTensor::Plain(lhs)) => zeros_grown(&call.client, lhs, key.m),
            _ => None,
        };
        // A grown lhs makes a product of another shape than the caller's buffer, so the round's
        // launches allocate their own.
        match grown {
            Some(lhs) => Self {
                lhs: MatmulCallInput::replaced_by(lhs),
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

    /// The input the product reads in place of this one: the float tensor `handle` holds.
    pub(crate) fn replaced_by(handle: TensorHandle) -> Self {
        Self::plain(PlainTensor::of_handle(handle))
    }
}

/// A zero copy of the dense float `lhs` with `rows` along its rows axis, laid out as `lhs` is: what
/// a round of a bucket of row counts measures. `None` where `lhs` already has as many rows, or is
/// not a dense buffer whose layout a grown one can repeat.
fn zeros_grown(client: &Client, lhs: &PlainTensor, rows: usize) -> Option<TensorHandle> {
    let meta = lhs.meta();
    let (shape, strides) = (meta.shape().as_slice(), &meta.strides()[..]);
    let axis = shape.len().checked_sub(2)?;
    if rows <= shape[axis] || lhs.is_tiled() {
        return None;
    }
    // The axes innermost first; a dense buffer's strides are the running product of the extents
    // inside them. An axis of one has no step to repeat.
    let mut order: Vec<usize> = (0..shape.len()).collect();
    order.sort_by_key(|&a| (strides[a], shape[a] != 1));
    let mut step = 1;
    for &a in &order {
        if shape[a] != 1 && strides[a] != step {
            return None;
        }
        step *= shape[a];
    }
    let mut grown = shape.to_vec();
    grown[axis] = rows;
    let mut grown_strides = vec![0; grown.len()];
    let mut step = 1;
    for &a in &order {
        grown_strides[a] = step;
        step *= grown[a];
    }
    let memory = client.create_from_slice(&vec![0u8; step * lhs.elem.size()]);
    Some(TensorHandle::new(
        memory,
        Shape::from(grown),
        grown_strides,
        lhs.elem,
    ))
}
