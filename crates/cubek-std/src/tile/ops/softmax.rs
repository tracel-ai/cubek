use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::mask::Mask;
use crate::tile::variants::InnerLayout;
use crate::tile::{Plane, RowWise, Tile, TileExpand, TileKind, TileKindExpand};

/// Logits below this are considered masked (effectively -inf).
/// Value chosen to fit within f16 range (~-65,504 max).
pub const LOGIT_MASKED: f32 = -6e4;

/// Row-shape descriptor for online softmax.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SoftmaxKind {
    /// `Tile::Unit` / `Tile::Register`: each unit owns its full tile.
    Direct { num_rows_per_unit: u32 },
    /// `Tile::WhiteboxFragment` / `Tile::Bounce`: plane-fragmented.
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

/// Initial `(m, l)` running state for online softmax.
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
impl<Acc: Float> Tile<Acc, Plane> {
    /// Online softmax update fused with the precision-cast write into the
    /// value-matmul lhs tile.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed_tile: &mut Tile<Lhs, Plane>,
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

    /// Copy `self` into `dest`.
    pub fn write_results<DE: Float, DS: Size>(&self, dest: &mut Tile<DE, Plane>) {
        dest.copy_from::<Acc, DS, Acc, Acc, Acc>(self, StageIdent::Out);
    }
}
