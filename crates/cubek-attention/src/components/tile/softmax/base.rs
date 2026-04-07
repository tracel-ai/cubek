use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::tile::StridedTile;

use crate::{components::tile::MaskTile, definition::AttentionTileSize};

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
    /// A `ScaleColumn` of scaling factors $\alpha_i = e^{m_{i, \text{old}} - m_{i, \text{new}}}$.
    fn softmax(
        score_matmul_accumulator: &mut Self::ScoreTile,
        mask: &MaskTile<F, Self>,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: F,
        #[comptime] softmax_config: Self::Config,
    ) -> Self::ScaleColumn;

    fn init_workspace(#[comptime] softmax_config: Self::Config) -> Self::Workspace;

    fn init_state(#[comptime] softmax_config: Self::Config) -> Self::RunningState;

    fn init_score_tile(#[comptime] config: Self::Config) -> Self::ScoreTile;
    fn zero_score_tile(score_tile: &mut Self::ScoreTile);

    fn init_softmax_tile(#[comptime] config: Self::Config) -> Self::SoftmaxedTile;

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask;
    fn load_mask<E: Numeric, ES: Size>(
        tile: &StridedTile<E, ES>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    );
    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout;
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
    fn num_rows_per_unit(&self) -> usize;
    fn tile_size(&self) -> AttentionTileSize;
}

#[cube]
/// Describes how a fragment is fragmented across units
/// The layout is independent of the data and data types
pub trait SoftmaxLayout: CubeType {
    /// Maps the (row, col) of the registers of a single unit to the position within the whole tile
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then we would have:
    /// unit_0: absolute_pos((0, 0)) == (0, 0)
    /// unit_0: absolute_pos((0, 1)) == (0, 1)
    /// unit_0: absolute_pos((1, 0)) == (2, 0)
    /// unit_0: absolute_pos((1, 1)) == (2, 1)
    /// ...
    /// unit_3: absolute_pos((0, 0)) == (1, 2)
    /// unit_3: absolute_pos((0, 1)) == (1, 3)
    /// unit_3: absolute_pos((1, 0)) == (3, 2)
    /// unit_3: absolute_pos((1, 1)) == (3, 3)
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d;

    /// Gives how many units participate in the same row
    ///
    /// Example: for simplicity, if we had a 4 units warp for a 4x4 tile divided as such:
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    ///  0, 0, 1, 1,
    ///  2, 2, 3, 3,
    /// Then it would output 2, because each row is spread across two different units (0 and 1, or 2 and 3)
    /// Layouts with varying num_units_per_row are not supported
    fn num_units_per_row(&self) -> comptime_type!(u32);
}

#[cube]
/// Describes which elements of a fragment should be masked
pub trait FragmentMask: CubeType {
    /// How the fragment is fragmented across units
    type Layout: SoftmaxLayout;

    /// Returns `true` if the element at `local_pos` should be masked
    fn should_mask(&self, local_pos: Coords2d) -> bool;
}
