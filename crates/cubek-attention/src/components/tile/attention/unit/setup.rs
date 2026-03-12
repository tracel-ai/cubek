use cubecl::ir::DeviceProperties;
use cubek_matmul::components::CubeDimResource;
use cubek_std::InvalidConfigError;

use crate::{
    components::tile::{
        SharedTileAttentionConfig, TileAttentionConfig, TileAttentionFamily,
        attention::unit::attention::UnitTileAttention, matmul::UnitMatmulConfig,
        output::unit::UnitOutputConfig, softmax::unit::UnitSoftmaxConfig,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        AttentionTileSize,
    },
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitTileAttentionConfig {
    pub shared: SharedTileAttentionConfig,
}

impl TileAttentionConfig for UnitTileAttentionConfig {
    type ScoreMatmulConfig = UnitMatmulConfig;
    type SoftmaxConfig = UnitSoftmaxConfig;
    type ValueMatmulConfig = UnitMatmulConfig;
    type AccumulatorConfig = UnitOutputConfig;

    fn score_matmul_config(&self) -> Self::ScoreMatmulConfig {
        todo!()
    }

    fn softmax_config(&self) -> Self::SoftmaxConfig {
        todo!()
    }

    fn value_matmul_config(&self) -> Self::ValueMatmulConfig {
        todo!()
    }

    fn accumulator_config(&self) -> Self::AccumulatorConfig {
        todo!()
    }

    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.shared.num_planes
    }

    fn tile_size(&self) -> AttentionTileSize {
        self.shared.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.shared.attention_tile_size.seq_q
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for UnitTileAttention {
    type TileAttention<F: AttentionPrecision> = UnitTileAttention;

    type Config = UnitTileAttentionConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Units(1))
    }

    fn expand_config(
        _device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        _dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        Ok(UnitTileAttentionConfig {
            shared: SharedTileAttentionConfig {
                plane_dim: blueprint.plane_dim,
                attention_tile_size: blueprint.tiling_scheme.tile_size,
                num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                causal_mask: blueprint.causal,
                materialized_mask: blueprint.masked,
            },
        })
    }
}
