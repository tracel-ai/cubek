use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::mask::Mask;
use crate::tile::variants::InnerLayout;
use crate::tile::{Plane, RowWise, Tile, TileExpand, TileKind, TileKindExpand};

/// Logits below this are considered masked (effectively -inf).
/// Value chosen to fit within f16 range (~-65,504 max).
pub const LOGIT_MASKED: f32 = -6e4;

/// Comptime descriptor for the row-shape used by online softmax. Determines
/// how many rows per unit each running-state vector holds.
///
/// - `Direct { num_rows_per_unit }` — used with `Tile::Unit` or `Tile::Register`
///   when each unit owns its own copy of the tile.
/// - `Plane { inner_layout }` — used with `Tile::WhiteboxFragment` or `Tile::Bounce`,
///   where the inner layout determines how many rows each unit covers.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SoftmaxKind {
    Direct { num_rows_per_unit: u32 },
    Plane { inner_layout: InnerLayout },
}

impl SoftmaxKind {
    pub const fn num_rows_per_unit(&self) -> u32 {
        match self {
            SoftmaxKind::Direct { num_rows_per_unit } => *num_rows_per_unit,
            SoftmaxKind::Plane { inner_layout } => match inner_layout {
                InnerLayout::Contiguous => 1,
                InnerLayout::SplitRows => 2,
            },
        }
    }
}

/// Initial running state `(m, l)` for the online softmax over a single tile row.
#[cube]
pub fn softmax_init_state<E: Float>(
    #[comptime] num_rows_per_unit: u32,
) -> (RowWise<E>, RowWise<E>) {
    (
        RowWise::<E>::new_min_value(num_rows_per_unit as usize),
        RowWise::<E>::new_zero(num_rows_per_unit as usize),
    )
}

#[cube]
impl<Acc: Float> Tile<Acc, Plane, ReadWrite> {
    /// Online softmax update over a single attention tile, fused with the
    /// precision-cast write into a value-matmul lhs tile. Each arm delegates
    /// to a `softmax` method on the variant's data struct.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed_tile: &mut Tile<Lhs, Plane, ReadWrite>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        match &mut self.kind {
            TileKind::Bounce(s) => {
                s.softmax::<Lhs, M>(mask, softmaxed_tile, state, head_dim_factor)
            }
            TileKind::WhiteboxFragment(s) => {
                s.softmax::<Lhs, M>(mask, softmaxed_tile, state, head_dim_factor)
            }
            TileKind::Unit(s) => s.softmax::<Lhs, M>(mask, softmaxed_tile, state, head_dim_factor),
            TileKind::Register(s) => {
                s.softmax::<Lhs, M>(mask, softmaxed_tile, state, head_dim_factor)
            }
            _ => panic!("softmax: unsupported score variant"),
        }
    }

    /// Copies `self` into `dest` (a stage-side strided/shared tile in the
    /// caller's downstream write path).
    pub fn write_results<DE: Float, DS: Size>(&self, dest: &mut Tile<DE, Plane, ReadWrite>) {
        dest.copy_from::<Acc, DS, Acc, Acc, Acc, ReadWrite>(self, StageIdent::Out);
    }
}
