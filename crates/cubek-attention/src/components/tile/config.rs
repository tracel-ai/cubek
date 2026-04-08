use crate::definition::AttentionTileSize;

use std::{fmt::Debug, hash::Hash};

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type ScoreMatmulConfig: Copy + Clone;
    type SoftmaxConfig: Copy + Clone;
    type ValueMatmulConfig: Copy + Clone;
    type AttentionOutputConfig: Copy + Clone;

    fn score_matmul_config(&self) -> Self::ScoreMatmulConfig;
    fn softmax_config(&self) -> Self::SoftmaxConfig;
    fn value_matmul_config(&self) -> Self::ValueMatmulConfig;
    fn output_config(&self) -> Self::AttentionOutputConfig;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn tile_size(&self) -> AttentionTileSize;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedTileAttentionConfig {
    pub plane_dim: u32,
    pub num_planes: u32,
    pub attention_tile_size: AttentionTileSize,
}
