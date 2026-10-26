use cubecl::client::Client;
use serde::{Deserialize, Serialize};

use crate::kernels::grouped_rows::{GroupedRows, PaddedRows, RouteSort};
use crate::kernels::tensor::PlainTensor;

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
    /// `silu(gate) · up` of another grouped product's `[gate | up]` output, `2 · k` wide and padded
    /// to its own tiles, over the same groups.
    Gated(PaddedRows),
}

/// Which [`MGroupedRowSource`] a product pads: a launch that gathers or gates its rows as it pads
/// them is timed apart from one that only copies them.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MGroupedRowSourceKind {
    /// Rows already in grouped order, copied as they are padded.
    Sorted,
    /// Rows gathered from their tokens through a route sort as they are padded.
    Routed,
    /// Rows gated out of another grouped product's output as they are padded.
    Gated,
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
            Self::Gated(both) => both.layout().rows(),
        }
    }

    /// The contraction each row carries.
    pub fn k(&self) -> usize {
        match self {
            Self::Sorted(rows) => rows.shape()[1],
            Self::Routed { tokens, .. } => tokens.shape()[1],
            Self::Gated(both) => both.rows().shape()[1] / 2,
        }
    }

    /// Which source this is, as the autotune key tells them apart.
    pub fn kind(&self) -> MGroupedRowSourceKind {
        match self {
            Self::Sorted(_) => MGroupedRowSourceKind::Sorted,
            Self::Routed { .. } => MGroupedRowSourceKind::Routed,
            Self::Gated(_) => MGroupedRowSourceKind::Gated,
        }
    }

    /// The tensor the rows are read out of.
    pub fn buffer(&self) -> PlainTensor {
        match self {
            Self::Sorted(rows) => rows.clone(),
            Self::Routed { tokens, .. } => tokens.clone(),
            Self::Gated(both) => PlainTensor::of_handle(both.rows().clone()),
        }
    }

    /// The rows padded under `ends` on `client`'s device, each group's cumulative end row as a
    /// `u32`, every group starting on a tile of `align` rows.
    pub fn pad(self, client: &Client, ends: PlainTensor, align: usize) -> PaddedRows {
        let layout = GroupedRows::new(client, ends, self.rows(), align);
        let rows = match self {
            Self::Sorted(rows) => layout.pad(rows),
            Self::Routed { tokens, sort } => layout.pad_routes(tokens, sort),
            Self::Gated(both) => layout.gate(&both),
        };
        PaddedRows::new(rows, layout)
    }
}
