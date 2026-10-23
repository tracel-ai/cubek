use crate::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use crate::kernels::{MATMUL_ELEMENT_UNVERIFIED, MatmulIndex};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Refusal;

/// A product's two operands, on one device and of one element the routines are
/// verified at: what every launch's tensors are read from, before a launch prepares them as it
/// reads them.
#[derive(Debug, Clone)]
pub struct MatmulInputs {
    lhs: PlainTensor,
    rhs: PlainTensor,
}

impl MatmulInputs {
    /// The float elements the routines are verified at: what a packed operand decodes into.
    pub const FLOATS: [ElemType; 3] = [
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
    ];

    /// The elements the routines are verified at, inputs and output alike: the floats, and `i32`,
    /// which sums in itself ([`MatmulRoutineProblem::sum`](crate::kernels::MatmulRoutineProblem::sum)).
    pub const VERIFIED: [ElemType; 4] = [
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::BF16),
        ElemType::Int(IntKind::I32),
    ];

    /// `lhs` and `rhs`, on `client`'s device, checked as every routine needs them.
    ///
    /// # Errors
    ///
    /// [`MATMUL_ELEMENT_UNVERIFIED`]: what the routines decline whatever the device, and
    /// [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does not run.
    pub fn new(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Refusal> {
        if lhs.elem != rhs.elem || !Self::VERIFIED.contains(&lhs.elem) {
            return Err(MATMUL_ELEMENT_UNVERIFIED);
        }
        if !lhs.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        Ok(Self { lhs, rhs })
    }

    /// Both operands where they lie, unless one aliases itself: what a launch tiling the rows
    /// reads, by strides.
    pub fn addressable(self, client: &Client) -> Self {
        Self {
            lhs: Self::without_self_aliasing(client, self.lhs),
            rhs: Self::without_self_aliasing(client, self.rhs),
        }
    }

    /// The operands, the lhs first: the only way out of the check, so no launch reads them
    /// unchecked.
    pub fn into_parts(self) -> (PlainTensor, PlainTensor) {
        (self.lhs, self.rhs)
    }

    /// `tensor` where no two of its indices address one element, a row-major copy of it where they do.
    ///
    /// Neither a pitch nor gaps between rows stop a read by strides; the runtime pitches rows whose
    /// bytes are not aligned, so a check on the buffer's size would copy those. A storage-tiled
    /// operand is read through its tiling, stored as a launch reads it, so it stands: a copy would
    /// lay it back in rows on every launch.
    fn without_self_aliasing(client: &Client, tensor: PlainTensor) -> PlainTensor {
        match tensor.is_tiled() || tensor.is_contiguous() || tensor.is_nonoverlapping() {
            true => tensor,
            false => tensor.into_contiguous(client),
        }
    }
}

/// A table a product's inputs read one entry of per batch, and what its entries say: `u32`
/// entries, its dims the product's batches.
#[derive(Debug, Clone)]
pub struct MatmulIndexTable {
    pub index: MatmulIndex,
    pub entries: PlainTensor,
}
