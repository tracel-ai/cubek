use crate::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, PlainTensor};
use crate::kernels::{MATMUL_ELEMENT_UNVERIFIED, MatmulIndex, MatmulTableBuffer, TileRows};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, IntKind};
use cubecl::tile::launch::Decline;

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
#[derive(Debug, Clone)]
pub enum MatmulIndexTable {
    /// Each batch's slab of a stack of `slabs` ([`MatmulIndex::Slab`]).
    Slab { slabs: usize, entries: PlainTensor },
    /// Each batch's run of `K`, as the ends of the runs ([`MatmulIndex::RunOfK`]).
    RunOfK { ends: PlainTensor },
    /// Each tile's rows of a flat lhs and output, and its slab ([`MatmulIndex::RowRunsOnSlab`]).
    RowRuns(Box<MatmulRowRuns>),
}

/// The tables of rows cut to runs: each tile's slab and where its rows end. Every run is at most
/// `tile_rows` long, which the launch does not check.
#[derive(Debug, Clone)]
pub struct MatmulRowRuns {
    pub slabs: usize,
    pub tile_rows: TileRows,
    pub entries: PlainTensor,
    pub runs: PlainTensor,
}

impl MatmulIndexTable {
    /// What the table says of each batch to the product.
    pub fn index(&self) -> MatmulIndex {
        match self {
            Self::Slab { slabs, .. } => MatmulIndex::Slab { slabs: *slabs },
            Self::RunOfK { .. } => MatmulIndex::RunOfK,
            Self::RowRuns(runs) => MatmulIndex::RowRunsOnSlab { slabs: runs.slabs },
        }
    }

    /// The entries, one per batch.
    pub fn entries(&self) -> &PlainTensor {
        match self {
            Self::Slab { entries, .. } => entries,
            Self::RunOfK { ends } => ends,
            Self::RowRuns(runs) => &runs.entries,
        }
    }

    /// The buffer `buffer` names.
    ///
    /// # Panics
    ///
    /// On [`MatmulTableBuffer::Runs`] of a table that cuts no rows to runs, which no index reads.
    pub fn buffer(&self, buffer: MatmulTableBuffer) -> &PlainTensor {
        match (buffer, self) {
            (MatmulTableBuffer::Entries, _) => self.entries(),
            (MatmulTableBuffer::Runs, Self::RowRuns(runs)) => &runs.runs,
            (MatmulTableBuffer::Runs, _) => panic!("only rows cut to runs hold runs"),
        }
    }
}
