use cubecl;
use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::pipeline::{
    BounceConfig, Mask, RowWise, RowwiseTileKind, RowwiseTileWorkspace, RowwiseTileWorkspaceExpand,
};
use crate::tile::{Plane, Tile, TileExpand};

/// Comptime configuration for [`SoftmaxWorkspace`]. Direct = each unit holds
/// the full tile in registers (no smem bounce). Bounce = the score and
/// softmaxed tiles round-trip through shared memory and a `LocalTile` view,
/// matching the cmma-style fragment layout.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SoftmaxKind {
    Direct { num_rows_per_unit: u32 },
    Bounce(BounceConfig),
}

#[derive(CubeType)]
pub struct SoftmaxWorkspace<Acc: Float, Lhs: Float> {
    pub max: RowWise<Acc>,
    pub sum: RowWise<Acc>,
    pub score: RowwiseTileWorkspace<Acc>,
    pub softmaxed: RowwiseTileWorkspace<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] kind: SoftmaxKind) -> SoftmaxWorkspace<Acc, Lhs> {
        let (rowwise_kind, num_rows_per_unit) = comptime! {
            match kind {
                SoftmaxKind::Direct { num_rows_per_unit } => {
                    (RowwiseTileKind::Direct, num_rows_per_unit as usize)
                }
                SoftmaxKind::Bounce(cfg) => {
                    let n = match cfg.inner_layout {
                        crate::tile::InnerLayout::Contiguous => 1usize,
                        crate::tile::InnerLayout::SplitRows => 2usize,
                    };
                    (RowwiseTileKind::Bounce(cfg), n)
                }
            }
        };

        SoftmaxWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(num_rows_per_unit),
            sum: RowWise::new_zero(num_rows_per_unit),
            score: RowwiseTileWorkspace::new(rowwise_kind),
            softmaxed: RowwiseTileWorkspace::new(rowwise_kind),
        }
    }
}

/// Initial running state `(m, l)` for the online softmax over a single tile row.
#[cube]
pub fn softmax_init_state<E: Float>(#[comptime] num_rows_per_unit: u32) -> (RowWise<E>, RowWise<E>) {
    (
        RowWise::<E>::new_min_value(num_rows_per_unit as usize),
        RowWise::<E>::new_zero(num_rows_per_unit as usize),
    )
}

#[cube]
impl<Acc: Float, V: Size> Tile<Acc, V, Plane, ReadWrite> {
    /// Online softmax update over a single attention tile, fused with the
    /// precision-cast write into a value-matmul lhs tile.
    ///
    /// Sequence of operations (all dispatched per-variant via the underlying
    /// PR3 primitives):
    /// 1. `bounce_in` — for `Tile::Cmma` only, populates `workspace.score.local_tile`.
    /// 2. `scale_and_mask`, `row_max`, `exp_diff`, `row_sum` on the score tile.
    /// 3. Online state update: `exp_m_diff`, `new_l`, write back into `state`.
    /// 4. Write the softmaxed values into `softmaxed_tile` with precision cast
    ///    (Register: direct array copy; Cmma: `local_tile -> softmaxed_smem -> cmma::load`).
    ///
    /// Returns the per-row scaling factor `α_i = e^(m_old - m_new)` used by the
    /// caller to rescale running output accumulators.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed_tile: &mut Tile<Lhs, Const<0>, Plane, ReadWrite>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        workspace: &mut SoftmaxWorkspace<Acc, Lhs>,
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        self.bounce_in(&mut workspace.score);

        self.scale_and_mask::<M>(head_dim_factor, mask, &mut workspace.score);
        self.row_max(&mut workspace.max, &state.0, &workspace.score);
        self.exp_diff(&workspace.max, &mut workspace.score);
        self.row_sum(&mut workspace.sum, &workspace.score);

        let exp_m_diff = state.0.exp_diff(&workspace.max);
        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        write_softmaxed(
            self,
            softmaxed_tile,
            &workspace.score,
            &mut workspace.softmaxed,
        );

        RowWise::copy_from(&mut state.0, &workspace.max);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }

    /// Multiplies each row of `self` by the corresponding `scale[r]` (cast into
    /// the tile's precision). For Cmma tiles this round-trips through smem.
    pub fn scale_mul<SM: Float>(
        &mut self,
        scale: &RowWise<SM>,
        workspace: &mut RowwiseTileWorkspace<Acc>,
    ) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        self.bounce_in(workspace);
        self.rowwise_scale(&scale_acc, workspace);
        self.bounce_out(workspace);
    }

    /// Divides each row of `self` by the corresponding `running_state_l[r]`,
    /// guarding against zero (a fully-masked row stays zero).
    pub fn scale_div<SM: Float>(
        &mut self,
        running_state_l: &RowWise<SM>,
        workspace: &mut RowwiseTileWorkspace<Acc>,
    ) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(running_state_l);
        scale.recip_inplace();
        self.bounce_in(workspace);
        self.rowwise_scale(&scale, workspace);
        self.bounce_out(workspace);
    }

    /// Copies `self` into `dest` (a stage-side strided/shared tile in the
    /// caller's downstream write path).
    pub fn write_results<DE: Float, DS: Size>(
        &self,
        dest: &mut Tile<DE, DS, Plane, ReadWrite>,
    ) {
        dest.copy_from::<Acc, V, Acc, Acc, Acc, ReadWrite>(self, StageIdent::Out);
    }
}

#[cube]
fn write_softmaxed<Acc: Float, V: Size, Lhs: Float>(
    score_tile: &Tile<Acc, V, Plane, ReadWrite>,
    softmaxed_tile: &mut Tile<Lhs, Const<0>, Plane, ReadWrite>,
    score_workspace: &RowwiseTileWorkspace<Acc>,
    softmaxed_workspace: &mut RowwiseTileWorkspace<Lhs>,
) {
    match (
        score_tile,
        softmaxed_tile,
        score_workspace,
        softmaxed_workspace,
    ) {
        (Tile::Register(s), Tile::Register(d), _, _) => {
            let m = comptime!(s.config.tile_size.m());
            let n = comptime!(s.config.tile_size.n());
            for i in 0..m * n {
                d.data[i as usize] = Lhs::cast_from(s.data[i as usize]);
            }
        }
        (
            Tile::Cmma(_),
            Tile::Cmma(d),
            RowwiseTileWorkspace::Bounce(score_bw),
            RowwiseTileWorkspace::Bounce(softmaxed_bw),
        ) => {
            let stride = comptime!(d.tile_size.n());
            score_bw.local_tile.store_to(&mut softmaxed_bw.smem);
            sync_cube();
            cmma::load(&d.matrix, &softmaxed_bw.smem.to_slice(), stride);
        }
        _ => panic!("write_softmaxed: incompatible tile/workspace combination"),
    }
}
