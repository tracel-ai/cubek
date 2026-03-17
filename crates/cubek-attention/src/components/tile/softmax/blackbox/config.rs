use crate::{
    components::tile::{pipeline::InnerLayout, softmax::SoftmaxConfig},
    definition::AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxSoftmaxConfig {
    pub tile_size: AttentionTileSize,
    pub plane_dim: u32,
    pub num_planes: u32,
    pub inner_layout: InnerLayout,
    pub causal_mask: bool,
    pub materialized_mask: bool,
}

impl SoftmaxConfig for BlackboxSoftmaxConfig {
    fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.materialized_mask
    }

    fn num_rows_per_unit(&self) -> usize {
        match self.inner_layout {
            InnerLayout::Contiguous => 1,
            InnerLayout::SplitRows => 2,
        }
    }

    fn tile_size(&self) -> AttentionTileSize {
        self.tile_size
    }
}
