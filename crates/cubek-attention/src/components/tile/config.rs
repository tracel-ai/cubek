use crate::definition::AttentionTileSize;

use std::fmt::Debug;
use std::hash::Hash;

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type ScoreMatmulConfig: Copy + Clone;
    type SoftmaxConfig: Copy + Clone;
    type ValueMatmulConfig: Copy + Clone;
    type AccumulatorConfig: Copy + Clone;

    fn score_matmul_config(&self) -> Self::ScoreMatmulConfig;
    fn softmax_config(&self) -> Self::SoftmaxConfig;
    fn value_matmul_config(&self) -> Self::ValueMatmulConfig;
    fn accumulator_config(&self) -> Self::AccumulatorConfig;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn tile_size(&self) -> AttentionTileSize;
    fn num_rows_per_unit(&self) -> u32;
    fn causal_mask(&self) -> bool;
    fn materialized_mask(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedTileAttentionConfig {
    pub plane_dim: u32,
    pub num_planes: u32,
    pub attention_tile_size: AttentionTileSize,
    pub causal_mask: bool,
    pub materialized_mask: bool,
}
