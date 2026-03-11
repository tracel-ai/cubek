use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::softmax::RowMax;
use crate::components::softmax::RowSum;
use crate::components::softmax::SoftmaxProcedureConfig;
use crate::components::softmax::base::Softmax;
use crate::components::stage::AttentionPartitioner;
use crate::components::tile::RowWise;
use crate::components::tile::SoftmaxRowwise;
use crate::components::tile::accelerated_blackbox::LocalTile;
use crate::components::tile::accelerated_blackbox::LocalTileLayout;

use crate::components::stage::MaskTile;
use crate::components::tile::SoftmaxRowwiseExpand;

#[derive(CubeType)]
pub struct SoftmaxProcedure<P: AttentionPartitioner, Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(P, Lhs)>,
}

#[derive(CubeType)]
pub struct SoftmaxProcedureWorkspace<Acc: Float, Lhs: Float> {
    max: RowWise<Acc>,
    sum: RowWise<Acc>,
    acc_smem_slice: SliceMut<Acc>,
    lhs_smem_slice: SliceMut<Lhs>,
    local_tile: LocalTile<Acc>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxProcedureWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: SoftmaxProcedureConfig) -> Self {
        SoftmaxProcedureWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit as usize),
            sum: RowWise::new_zero(config.num_rows_per_unit as usize),
            // Create smem and slice for this plane
            acc_smem_slice: todo!(),
            // Create smem and slice for this plane
            lhs_smem_slice: todo!(),
            local_tile: LocalTile::new(LocalTileLayout::new(
                (config.tile_size.seq_q, config.tile_size.seq_kv),
                config.plane_dim,
                config.inner_layout,
            )),
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float, P: AttentionPartitioner> Softmax<Acc> for SoftmaxProcedure<P, Lhs> {
    type Config = SoftmaxProcedureConfig;
    type ScaleColumn = RowWise<Acc>;
    type RunningState = (RowWise<Acc>, RowWise<Acc>);
    type ScoreTile = cmma::Matrix<Acc>;
    type SoftmaxedTile = cmma::Matrix<Lhs>;
    type Workspace = SoftmaxProcedureWorkspace<Acc, Lhs>;
    type Mask = MaskTile<Acc, Self>;
    type ScoreLayout = LocalTileLayout;

    fn softmax(
        score_matmul_accumulator: &Self::ScoreTile,
        mask: &Self::Mask,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: Acc,
        #[comptime] config: Self::Config,
    ) -> Self::ScaleColumn {
        // Make sure the softmax is in a row-aware layout
        // If the layout is always row-aware, it's a no-op.
        // Otherwise it may go through shared memory
        // let softmax_rowwise = value_matmul_lhs.rowwise_mut();
        cmma::store(
            &mut workspace.acc_smem_slice,
            &score_matmul_accumulator,
            config.tile_size.seq_kv,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.acc_smem_slice.to_slice());

        sync_cube();

        // Perform the softmax calculation on the (row-format) softmax tile, including masking
        // This mutates the (row-format) softmax tile and the state
        // Also outputs a value needed to scale accumulator later
        let scale = tile_softmax::<Acc, Self, LocalTile<Acc>, P::Reducer>(
            &mut workspace.local_tile,
            mask,
            &mut state.0,
            &mut state.1,
            &mut workspace.max,
            &mut workspace.sum,
            head_dim_factor,
        );

        // Make sure the mutations on softmax_rowwise also affect other softmax formats
        workspace.local_tile.store_to(&mut workspace.lhs_smem_slice);

        sync_cube();

        cmma::load(
            &value_matmul_lhs,
            &workspace.lhs_smem_slice.to_slice(),
            config.tile_size.seq_kv,
        );

        scale
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::RunningState {
        (
            RowWise::<Acc>::new_min_value(config.num_rows_per_unit as usize),
            RowWise::<Acc>::new_zero(config.num_rows_per_unit as usize),
        )
    }

    fn init_score_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) -> Self::ScoreTile {
        unsafe {
            cmma::Matrix::<Acc>::uninitialized(
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
        #[comptime] config: Self::Config,
    ) -> Self::SoftmaxedTile {
        unsafe {
            cmma::Matrix::<Lhs>::uninitialized(
                cmma::MatrixIdent::A,
                config.tile_size.seq_q as usize,
                config.tile_size.seq_kv as usize,
                config.tile_size.head_dim as usize,
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        todo!()
    }
    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout {
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
pub fn tile_softmax<F: Float, SMX: Softmax<F>, SR: SoftmaxRowwise<F>, R: Reducer>(
    rowwise_softmax: &mut SR,
    mask: &MaskTile<F, SMX>,
    state_m: &mut RowWise<F>,
    state_l: &mut RowWise<F>,
    max_placeholder: &mut RowWise<F>,
    sum_placeholder: &mut RowWise<F>,
    head_dim_factor: F,
) -> RowWise<F> {
    SR::scale_and_mask::<MaskTile<F, SMX>>(rowwise_softmax, head_dim_factor, mask);

    row_max::<F, SR, R>(max_placeholder, state_m, rowwise_softmax);

    rowwise_softmax.exp_diff(max_placeholder);

    row_sum::<F, SR, R>(sum_placeholder, rowwise_softmax);

    let exp_m_diff = state_m.exp_diff(max_placeholder);

    let new_l = exp_m_diff.mul(state_l).add(sum_placeholder);

    RowWise::copy_from(state_m, max_placeholder);
    RowWise::copy_from(state_l, &new_l);

    exp_m_diff
}

#[cube]
/// Computes the sum of rows on a fragment, using the Reducer's strategy
pub fn row_sum<E: Float, F: SoftmaxRowwise<E>, R: Reducer>(vals: &mut RowWise<E>, data: &F) {
    vals.fill(E::from_int(0));
    R::reduce::<E, F, RowSum>(vals, data)
}

#[cube]
/// Computes the max of rows on a fragment, using the Reducer's strategy
/// Starts max at base
pub fn row_max<E: Float, F: SoftmaxRowwise<E>, R: Reducer>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &F,
) {
    vals.copy_from(base);
    R::reduce::<E, F, RowMax>(vals, data)
}

#[cube]
/// Strategy for reducing across units participating in the same row
pub trait Reducer: CubeType {
    /// Reduction algorithm, applied inplace in vals
    fn reduce<E: Float, F: SoftmaxRowwise<E>, RO: ReduceOp<E>>(vals: &mut RowWise<E>, data: &F);
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
