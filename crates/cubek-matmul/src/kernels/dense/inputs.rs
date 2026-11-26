use crate::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use crate::kernels::{MATMUL_ELEMENT_UNVERIFIED, MatmulIndex};
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::Indexing;
use cubecl::tile::launch::Decline;
use cubek_std::matmul_dimensions::{M, MI};

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
    pub fn new(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Decline> {
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
            lhs: self.lhs.addressable(client),
            rhs: self.rhs.addressable(client),
        }
    }

    /// The operands, the lhs first: the only way out of the check, so no launch reads them
    /// unchecked.
    pub fn into_parts(self) -> (PlainTensor, PlainTensor) {
        (self.lhs, self.rhs)
    }
}

/// A table a product's inputs read one entry of per batch, and what its entries say: `u32`
/// entries, its dims the product's batches.
///
/// Under [`MatmulIndex::RunOfMOnSlab`] the table also carries the ends of the runs of rows. Every
/// run must be at most `rows_per_run` rows: a longer run leaves its later output rows unwritten.
#[derive(Debug, Clone)]
pub struct MatmulIndexTable {
    pub index: MatmulIndex,
    pub entries: PlainTensor,
    pub row_ends: Option<RowEnds>,
}

/// Where each batch's run of rows ends, the one before it being where it starts, and the most
/// rows any run holds.
#[derive(Debug, Clone)]
pub struct RowEnds {
    pub ends: PlainTensor,
    pub rows_per_run: NonZeroUsize,
}

impl MatmulIndexTable {
    /// The buffer an operand placed by `indexing` reads: the ends for a run of the rows, the
    /// entries for any other placement.
    pub fn read_by(&self, indexing: &Indexing) -> &PlainTensor {
        match (&self.row_ends, indexing) {
            (Some(row_ends), Indexing::Run(rows)) if [M, MI].contains(rows) => &row_ends.ends,
            _ => &self.entries,
        }
    }
}
