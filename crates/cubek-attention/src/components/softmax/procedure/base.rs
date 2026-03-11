use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::softmax::InnerMatmul;
use crate::components::softmax::RowMax;
use crate::components::softmax::RowSum;
use crate::components::softmax::TileAttention;
use crate::components::softmax::base::Softmax;
use crate::components::softmax::base::SoftmaxConfig;
use crate::components::stage::AttentionPartitioner;
use crate::components::tile::RowWise;
use crate::components::tile::SoftmaxRowwise;
use crate::components::tile::accelerated_blackbox::LocalTileLayout;
use crate::components::tile::{SoftmaxPipeline, SoftmaxPipelineExpand};

use crate::components::stage::MaskTile;
use crate::components::tile::SoftmaxRowwiseExpand;

use crate::definition::AttentionPrecision;
use crate::definition::attention_types::SM;
use crate::definition::attention_types::SML;

#[derive(CubeType)]
pub struct SoftmaxProcedure<AP: AttentionPrecision, TA: TileAttention<AP>, P: AttentionPartitioner>
{
    #[cube(comptime)]
    _phantom: PhantomData<(AP, TA, P)>,
}

#[derive(CubeType)]
pub struct SoftmaxProcedureWorkspace<Acc: Float, Lhs: Float> {
    max: RowWise<Acc>,
    sum: RowWise<Acc>,
    acc_smem: SharedMemory<Acc>,
    lhs_smem: SharedMemory<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxProcedureWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: SoftmaxConfig) -> Self {
        SoftmaxProcedureWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit as usize),
            sum: RowWise::new_zero(config.num_rows_per_unit as usize),
            acc_smem: todo!(),
            lhs_smem: todo!(),
        }
    }
}

#[cube]
impl<
    AP: AttentionPrecision,
    TA: TileAttention<
        AP,
        // SoftmaxRow = <BlackboxSoftmaxPipeline<SM<AP>, SML<AP>> as SoftmaxPipeline<SM<AP>>>::Rowwise,
    >,
    P: AttentionPartitioner,
> Softmax<SM<AP>> for SoftmaxProcedure<AP, TA, P>
{
    type ScaleColumn = RowWise<SM<AP>>;
    type RunningState = (RowWise<SM<AP>>, RowWise<SM<AP>>);
    type ScoreTile = cmma::Matrix<SM<AP>>;
    type SoftmaxedTile = cmma::Matrix<SML<AP>>;
    type Workspace = SoftmaxProcedureWorkspace<SM<AP>, SML<AP>>;
    type Mask = MaskTile<SM<AP>, Self>;
    type ScoreLayout = LocalTileLayout;

    fn softmax(
        score_matmul_accumulator: &Self::ScoreTile,
        mask: &Self::Mask,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: SM<AP>,
        #[comptime] softmax_config: SoftmaxConfig,
    ) -> Self::ScaleColumn {
        // Note: we use value_matmul_lhs because it's the mut one

        // Make sure the softmax is in a row-aware layout
        // If the layout is always row-aware, it's a no-op.
        // Otherwise it may go through shared memory
        let softmax_rowwise = value_matmul_lhs.rowwise_mut();

        // Perform the softmax calculation on the (row-format) softmax tile, including masking
        // This mutates the (row-format) softmax tile and the state
        // Also outputs a value needed to scale accumulator later
        let scale = tile_softmax::<AP, TA, P::Reducer>(
            softmax_rowwise,
            mask,
            &mut state.0,
            &mut state.1,
            &mut workspace.max,
            &mut workspace.sum,
            head_dim_factor,
            softmax_config,
        );

        // Make sure the mutations on softmax_rowwise also affect other softmax formats
        value_matmul_lhs.finalize_lhs();

        scale
    }

    fn init_workspace(#[comptime] config: SoftmaxConfig) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_state(#[comptime] config: SoftmaxConfig) -> Self::RunningState {
        (
            RowWise::<SM<AP>>::new_min_value(config.num_rows_per_unit() as usize),
            RowWise::<SM<AP>>::new_zero(config.num_rows_per_unit() as usize),
        )
    }

    fn init_score_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: SoftmaxConfig,
    ) -> Self::ScoreTile {
        unsafe {
            cmma::Matrix::<SM<AP>>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                config.tile_size.seq_q as usize,
                config.tile_size.seq_kv as usize,
                config.tile_size.head_dim as usize,
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {}

    fn init_softmax_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: SoftmaxConfig,
    ) -> Self::SoftmaxedTile {
        unsafe {
            cmma::Matrix::<SML<AP>>::uninitialized(
                cmma::MatrixIdent::A,
                config.tile_size.seq_q as usize,
                config.tile_size.seq_kv as usize,
                config.tile_size.head_dim as usize,
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn allocate_mask(#[comptime] config: SoftmaxConfig) -> Self::Mask {
        todo!()
    }
    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] config: SoftmaxConfig,
    ) {
        todo!()
    }
    fn layout(#[comptime] config: SoftmaxConfig) -> Self::ScoreLayout {
        todo!()
    }
}

#[cube]
/// Applies softmax to a tile with masking and updates the running state.
///
/// Scales by `1 / sqrt(head_dim)`, applies the mask, computes row-wise max and sum,
/// exponentiates, and updates the softmax state.
///
/// Returns the exponential difference used for normalization.
pub fn tile_softmax<AP: AttentionPrecision, TA: TileAttention<AP>, R: Reducer>(
    rowwise_softmax: &<<TA as TileAttention<AP>>::ScoreMatmul as InnerMatmul>::Acc,
    mask: &MaskTile<AP, TA>,
    state_m: &mut RowWise<SM<AP>>,
    state_l: &mut RowWise<SM<AP>>,
    max_placeholder: &mut RowWise<SM<AP>>,
    sum_placeholder: &mut RowWise<SM<AP>>,
    head_dim_factor: SM<AP>,
    #[comptime] config: SoftmaxConfig,
) -> RowWise<SM<AP>> {
    TA::SoftmaxRow::scale_and_mask::<MaskTile<AP, TA>>(rowwise_softmax, head_dim_factor, mask);

    row_max::<SM<AP>, <<TA as TileAttention<AP>>::ScoreMatmul as InnerMatmul>::Acc, R>(
        max_placeholder,
        state_m,
        rowwise_softmax,
        config,
    );

    rowwise_softmax.exp_diff(max_placeholder);

    row_sum::<SM<AP>, <<TA as TileAttention<AP>>::ScoreMatmul as InnerMatmul>::Acc, R>(
        sum_placeholder,
        rowwise_softmax,
        config,
    );

    let exp_m_diff = state_m.exp_diff(max_placeholder);

    let new_l = exp_m_diff.mul(state_l).add(sum_placeholder);

    RowWise::copy_from(state_m, max_placeholder);
    RowWise::copy_from(state_l, &new_l);

    exp_m_diff
}

#[cube]
/// Computes the sum of rows on a fragment, using the Reducer's strategy
pub fn row_sum<E: Float, F: SoftmaxRowwise<E>, R: Reducer>(
    vals: &mut RowWise<E>,
    data: &F,
    #[comptime] config: SoftmaxConfig,
) {
    vals.fill(E::from_int(0));
    R::reduce::<E, F, RowSum>(vals, data, config)
}

#[cube]
/// Computes the max of rows on a fragment, using the Reducer's strategy
/// Starts max at base
pub fn row_max<E: Float, F: SoftmaxRowwise<E>, R: Reducer>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &F,
    #[comptime] config: SoftmaxConfig,
) {
    vals.copy_from(base);
    R::reduce::<E, F, RowMax>(vals, data, config)
}

#[cube]
/// Strategy for reducing across units participating in the same row
pub trait Reducer: CubeType {
    /// Reduction algorithm, applied inplace in vals
    fn reduce<E: Float, F: SoftmaxRowwise<E>, RO: ReduceOp<E>>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] config: SoftmaxConfig,
    );
}

#[cube]
/// A reduction operation
pub trait ReduceOp<E: Float> {
    /// Applies the reduction on the elements of the same row held by the unit
    fn reduce_local<F: SoftmaxRowwise<E>>(data: &F) -> RowWise<E>;

    /// Applies the reduction on the elements of the same row held by the unit,
    /// and to the accumulator, and store in the accumulator
    fn reduce_local_accumulate<F: SoftmaxRowwise<E>>(data: &F, acc: &mut RowWise<E>);

    /// The basic operation on two single values
    #[allow(unused)]
    fn reduce_step_scalar(a: E, b: E) -> E;

    /// Accumulates elem into acc.
    /// If mask is activated, the element gets masked prior to being accumulated
    fn reduce_step_rowwise(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}
