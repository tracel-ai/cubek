use cubecl;
use cubecl::prelude::*;
use cubek_std::TileSize;
use cubek_std::tile::StridedTile;

use crate::components::stage::MaskTile;
use crate::components::tile::accelerated_blackbox::InnerLayout;
use crate::components::tile::{FragmentMask, SoftmaxLayout, TileAttentionConfig};
use crate::definition::attention_types::SM;
use crate::definition::{AttentionPrecision, AttentionTileSize};

#[cube]
pub trait InnerMatmul {
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Acc: CubeType;
    type Config: Copy + Clone;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;
    fn load_lhs<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Lhs);

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;
    fn load_rhs_plain<E: Float>(tile: &StridedTile<E>, fragment: &mut Self::Rhs);
    fn load_rhs_transposed<E: Float>(tile: &StridedTile<E>, fragment: &mut Self::Rhs);

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] tile_size: TileSize,
    );
}

#[cube]
pub trait Softmax<F: Float>: Send + Sync + 'static + Sized {
    /// Vector type representing one entry per row of a fragment.
    /// Used for row-wise statistics (max, sum, scaling factors).
    type ScaleColumn: CubeType;

    type RunningState: CubeType;

    /// The input tile containing raw attention scores (typically higher precision),
    /// from which softmax calculations take their inputs
    type ScoreTile: CubeType;

    /// The output tile containing normalized probabilities,
    /// formatted for immediate use as the LHS in Value MatMul.
    type SoftmaxedTile: CubeType;

    /// Implementation-defined temporary storage (e.g., register placeholders)
    /// to be reused across iterations to minimize register pressure.
    type Workspace: CubeType;

    type Mask: FragmentMask<Layout = Self::ScoreLayout>;
    type ScoreLayout: SoftmaxLayout;
    type Config: SoftmaxConfig;

    /// Executes the online softmax update and layout transformation.
    ///
    /// 1. Scales and masks the `score_matmul_accumulator`.
    /// 2. Updates the running row-wise statistics (`state_m` and `state_l`).
    /// 3. Computes exponentials and normalizes values.
    /// 4. Transforms and casts the result into `value_matmul_lhs`.
    ///
    /// # Returns
    /// A `OutColumn` of scaling factors $\alpha_i = e^{m_{i, \text{old}} - m_{i, \text{new}}}$.
    fn softmax(
        score_matmul_accumulator: &Self::ScoreTile,
        mask: &MaskTile<F, Self>,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: F,
        #[comptime] softmax_config: Self::Config,
    ) -> Self::ScaleColumn;

    fn init_workspace(#[comptime] softmax_config: Self::Config) -> Self::Workspace;

    fn init_state(#[comptime] softmax_config: Self::Config) -> Self::RunningState;

    fn init_score_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) -> Self::ScoreTile;

    fn zero_score_tile(score_tile: &mut Self::ScoreTile);

    fn init_softmax_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) -> Self::SoftmaxedTile;

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask;
    fn load_mask<E: Numeric>(
        tile: &StridedTile<E>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    );
    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout;
}

#[cube]
pub trait Accumulator: Send + Sync + 'static + Sized {
    type Config: Copy + Clone;
    type ScaleColumn: CubeType;
    type RunningState: CubeType;
    type Tile: CubeType;
    type Workspace: CubeType;

    fn scale_mul(tile: &mut Self::Tile, column: &Self::ScaleColumn);
    fn scale_div(tile: &mut Self::Tile, running_state: &Self::RunningState);

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace;

    fn init_tile(workspace: &mut Self::Workspace, #[comptime] config: Self::Config) -> Self::Tile;

    fn write_results<E: Float>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    );
}

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: TileAttentionConfig<
            ScoreMatmulConfig = <Self::ScoreMatmul as InnerMatmul>::Config,
            ValueMatmulConfig = <Self::ValueMatmul as InnerMatmul>::Config,
            AccumulatorConfig = <Self::Accumulator as Accumulator>::Config,
        >;
    type ScoreMatmul: InnerMatmul;
    type Softmax: Softmax<
            SM<AP>,
            ScoreTile = <Self::ScoreMatmul as InnerMatmul>::Acc,
            SoftmaxedTile = <Self::ValueMatmul as InnerMatmul>::Lhs,
            ScaleColumn = <Self::Accumulator as Accumulator>::ScaleColumn,
            RunningState = <Self::Accumulator as Accumulator>::RunningState,
            Config = <Self::Config as TileAttentionConfig>::SoftmaxConfig,
        >;
    type ValueMatmul: InnerMatmul;
    type Accumulator: Accumulator<Tile = <Self::ValueMatmul as InnerMatmul>::Acc>;
}

pub trait SoftmaxConfig: Copy + Clone {
    // pub num_rows_per_unit: u32,
    // pub plane_dim: u32,
    // pub num_planes: u32,
    // pub tile_size: AttentionTileSize,
    // pub causal_mask: bool,
    // pub materialized_mask: bool,

    fn causal_mask(&self) -> bool;
    fn materialized_mask(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SoftmaxProcedureConfig {
    pub num_rows_per_unit: usize,
    pub tile_size: AttentionTileSize,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

impl SoftmaxConfig for SoftmaxProcedureConfig {
    fn causal_mask(&self) -> bool {
        todo!()
    }

    fn materialized_mask(&self) -> bool {
        todo!()
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AccumulatorConfig {}

impl AccumulatorConfig {}
