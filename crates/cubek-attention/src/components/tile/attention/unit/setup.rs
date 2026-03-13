use cubecl::ir::DeviceProperties;
use cubek_matmul::components::CubeDimResource;
use cubek_std::InvalidConfigError;

use crate::{
    components::tile::{
        SharedTileAttentionConfig, TileAttentionConfig, TileAttentionFamily,
        attention::unit::attention::UnitTileAttention, matmul::UnitMatmulConfig,
        output::unit::UnitOutputConfig, pipeline::InnerLayout, softmax::unit::UnitSoftmaxConfig,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        AttentionTileSize,
    },
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitTileAttentionConfig {
    pub shared: SharedTileAttentionConfig,
    pub score_matmul_config: UnitMatmulConfig,
    pub softmax_config: UnitSoftmaxConfig,
    pub value_matmul_config: UnitMatmulConfig,
    pub output_config: UnitOutputConfig,
}

impl TileAttentionConfig for UnitTileAttentionConfig {
    type ScoreMatmulConfig = UnitMatmulConfig;
    type SoftmaxConfig = UnitSoftmaxConfig;
    type ValueMatmulConfig = UnitMatmulConfig;
    type AttentionOutputConfig = UnitOutputConfig;

    fn score_matmul_config(&self) -> Self::ScoreMatmulConfig {
        self.score_matmul_config
    }

    fn softmax_config(&self) -> Self::SoftmaxConfig {
        self.softmax_config
    }

    fn value_matmul_config(&self) -> Self::ValueMatmulConfig {
        self.value_matmul_config
    }

    fn output_config(&self) -> Self::AttentionOutputConfig {
        self.output_config
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
        let plane_dim = blueprint.plane_dim;
        let attention_tile_size = blueprint.tiling_scheme.tile_size;
        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let inner_layout = if blueprint.two_rows_in_array_tile {
            InnerLayout::SplitRows
        } else {
            InnerLayout::Contiguous
        };
        Ok(UnitTileAttentionConfig {
            shared: SharedTileAttentionConfig {
                plane_dim,
                attention_tile_size,
                num_planes,
            },
            score_matmul_config: UnitMatmulConfig {
                tile_size: blueprint
                    .tiling_scheme
                    .tile_size
                    .to_score_matmul_tile_size(),
            },
            softmax_config: UnitSoftmaxConfig {
                tile_size: attention_tile_size,
                plane_dim,
                num_planes,
                inner_layout,
                causal_mask: blueprint.causal,
                materialized_mask: blueprint.masked,
            },
            value_matmul_config: UnitMatmulConfig {
                tile_size: blueprint
                    .tiling_scheme
                    .tile_size
                    .to_value_matmul_tile_size(),
            },
            output_config: UnitOutputConfig {
                tile_size: blueprint.tiling_scheme.tile_size,
            },
        })
    }
}
