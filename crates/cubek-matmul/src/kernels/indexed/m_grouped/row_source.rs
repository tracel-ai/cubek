use crate::kernels::indexed::m_grouped::{GroupedRows, PaddedRows, RouteSort};
use crate::kernels::tensor::PlainTensor;
use cubecl::client::Client;
use serde::{Deserialize, Serialize};

/// Where a grouped product's rows come from: what its launch pads into tiles a cube high before
/// the product reads them, each source in the pass it makes anyway.
#[derive(Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "built once per launch and moved into it: boxing the wide variant would add an \
              allocation per launch to save a copy on the stack"
)]
pub enum MGroupedRowSource {
    /// `[rows, k]`, already in grouped order.
    Sorted(PlainTensor),
    /// The rows `sort` placed, read out of their tokens, `[tokens, k]`.
    Routed {
        tokens: PlainTensor,
        sort: RouteSort,
    },
    /// `activation` of another grouped product's output, padded to its own tiles, over the same
    /// groups.
    Activated {
        rows: PaddedRows,
        activation: RowActivation,
    },
}

/// The activation an [`MGroupedRowSource::Activated`] source applies to each row as it moves the
/// row between two products' tiles.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RowActivation {
    /// `silu(gate) · up` of a `[gate | up]` row, half as wide.
    SiluGated,
    /// `max(x, 0)²`, as wide as the row.
    ReluSquared,
}

impl RowActivation {
    /// How many expansions of the hidden width a row holds before this activation.
    pub fn expansions(self) -> usize {
        match self {
            Self::SiluGated => 2,
            Self::ReluSquared => 1,
        }
    }

    /// The width of a row this activation makes out of one `width` wide.
    pub fn activated_width(self, width: usize) -> usize {
        width / self.expansions()
    }
}

/// Which [`MGroupedRowSource`] a product pads: a launch that gathers or activates its rows as it pads
/// them is timed apart from one that only copies them.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MGroupedRowSourceKind {
    /// Rows already in grouped order, copied as they are padded.
    Sorted,
    /// Rows gathered from their tokens through a route sort as they are padded.
    Routed,
    /// Rows activated out of another grouped product's output as they are padded.
    Activated(RowActivation),
}

impl From<PlainTensor> for MGroupedRowSource {
    fn from(rows: PlainTensor) -> Self {
        Self::Sorted(rows)
    }
}

impl MGroupedRowSource {
    /// How many rows are grouped.
    pub fn rows(&self) -> usize {
        match self {
            Self::Sorted(rows) => rows.shape()[0],
            Self::Routed { sort, .. } => sort.route_at_place.shape().num_elements(),
            Self::Activated { rows, .. } => rows.layout().rows(),
        }
    }

    /// The contraction each row carries.
    pub fn k(&self) -> usize {
        match self {
            Self::Sorted(rows) => rows.shape()[1],
            Self::Routed { tokens, .. } => tokens.shape()[1],
            Self::Activated { rows, activation } => {
                activation.activated_width(rows.rows().shape()[1])
            }
        }
    }

    /// Which source this is, as the autotune key tells them apart.
    pub fn kind(&self) -> MGroupedRowSourceKind {
        match self {
            Self::Sorted(_) => MGroupedRowSourceKind::Sorted,
            Self::Routed { .. } => MGroupedRowSourceKind::Routed,
            Self::Activated { activation, .. } => MGroupedRowSourceKind::Activated(*activation),
        }
    }

    /// The tensor the rows are read out of.
    pub fn buffer(&self) -> PlainTensor {
        match self {
            Self::Sorted(rows) => rows.clone(),
            Self::Routed { tokens, .. } => tokens.clone(),
            Self::Activated { rows, .. } => PlainTensor::of_handle(rows.rows().clone()),
        }
    }

    /// The rows padded under `ends` on `client`'s device, each group's cumulative end row as a
    /// `u32`, every group starting on a tile of `align` rows.
    pub fn pad(self, client: &Client, ends: PlainTensor, align: usize) -> PaddedRows {
        let layout = GroupedRows::new(client, ends, self.rows(), align);
        let rows = match self {
            Self::Sorted(rows) => layout.pad(rows),
            Self::Routed { tokens, sort } => layout.pad_routes(tokens, sort),
            Self::Activated { rows, activation } => layout.activate(&rows, activation),
        };
        PaddedRows::new(rows, layout)
    }
}
