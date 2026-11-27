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

/// A table a product's inputs read one entry of per batch, `u32` entries whose dims are the
/// product's batches, and how they place the inputs.
#[derive(Debug, Clone)]
pub enum MatmulIndexTable {
    /// Each entry names the slab of a stack of `slabs` the batch reads ([`MatmulIndex::Slab`]).
    Slab { slabs: usize, entries: PlainTensor },
    /// Each entry is where the batch's run of `K` ends ([`MatmulIndex::RunOfK`]).
    RunOfK { ends: PlainTensor },
    /// Each entry names a slab, and `groups` where each batch's row group ends
    /// ([`MatmulIndex::GroupedRows`]).
    GroupedRows {
        slabs: usize,
        entries: PlainTensor,
        groups: RowGroups,
    },
}

/// Where each batch's row group ends, nondecreasing and at most the flat rows, the one before it
/// being where it starts; and the most rows any row group holds, past which its later output rows
/// are left unwritten.
#[derive(Debug, Clone)]
pub struct RowGroups {
    pub ends: PlainTensor,
    pub max_rows: NonZeroUsize,
}

impl MatmulIndexTable {
    /// How the entries place the inputs.
    pub fn index(&self) -> MatmulIndex {
        match self {
            Self::Slab { slabs, .. } => MatmulIndex::Slab { slabs: *slabs },
            Self::RunOfK { .. } => MatmulIndex::RunOfK,
            Self::GroupedRows { slabs, .. } => MatmulIndex::GroupedRows { slabs: *slabs },
        }
    }

    /// The table read one entry of per batch.
    pub fn entries(&self) -> &PlainTensor {
        match self {
            Self::Slab { entries, .. } | Self::GroupedRows { entries, .. } => entries,
            Self::RunOfK { ends } => ends,
        }
    }

    /// The row groups, under [`MatmulIndex::GroupedRows`].
    pub fn groups(&self) -> Option<&RowGroups> {
        match self {
            Self::GroupedRows { groups, .. } => Some(groups),
            Self::Slab { .. } | Self::RunOfK { .. } => None,
        }
    }

    /// The buffer an operand placed by `indexing` reads: the ends for a row group, the
    /// entries for any other placement.
    pub fn read_by(&self, indexing: &Indexing) -> &PlainTensor {
        match (self, indexing) {
            (Self::GroupedRows { groups, .. }, Indexing::Run(rows)) if [M, MI].contains(rows) => {
                &groups.ends
            }
            _ => self.entries(),
        }
    }
}
