use crate::components::tile::matmul::AttentionTileMatmul;
use crate::definition::AttentionTileSize;

use std::{fmt::Debug, hash::Hash};

/// Configuration for the Tile Attention level
pub trait TileAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type SoftmaxConfig: Copy + Clone;
    type AttentionOutputConfig: Copy + Clone;

    fn score_matmul(&self) -> AttentionTileMatmul;
    fn value_matmul(&self) -> AttentionTileMatmul;
    fn softmax_config(&self) -> Self::SoftmaxConfig;
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
