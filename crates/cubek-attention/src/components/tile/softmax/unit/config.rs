use crate::{
    components::tile::{pipeline::InnerLayout, softmax::SoftmaxConfig},
    definition::AttentionTileSize,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitSoftmaxConfig {
    pub num_rows_per_unit: usize,
    pub tile_size: AttentionTileSize,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

impl SoftmaxConfig for UnitSoftmaxConfig {
    fn causal_mask(&self) -> bool {
        todo!()
    }

    fn materialized_mask(&self) -> bool {
        todo!()
    }
}
