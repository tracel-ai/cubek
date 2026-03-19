use cubecl::ir::DeviceProperties;
use cubecl::ir::VectorSize;
use cubek_matmul::components::CubeDimResource;
use cubek_matmul::definition::MatmulAvailabilityError;
use cubek_std::InvalidConfigError;

use crate::components::tile::SharedTileAttentionConfig;
use crate::components::tile::TileAttentionConfig;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::attention::blackbox::attention::BlackboxAcceleratedTileAttention;
use crate::components::tile::matmul::CmmaMatmulConfig;
use crate::components::tile::output::blackbox::BlackboxOutputConfig;
use crate::components::tile::pipeline::InnerLayout;
use crate::components::tile::softmax::blackbox::BlackboxSoftmaxConfig;
use crate::definition::AttentionAvailabilityError;
use crate::definition::AttentionBlueprint;
use crate::definition::AttentionElems;
use crate::definition::AttentionPrecision;
use crate::definition::AttentionSetupError;
use crate::definition::AttentionTileSize;
use cubecl::features::MmaConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxAcceleratedAttentionConfig {
    pub shared: SharedTileAttentionConfig,
    pub inner_layout: InnerLayout,
    pub score_matmul_config: CmmaMatmulConfig,
    pub softmax_config: BlackboxSoftmaxConfig,
    pub value_matmul_config: CmmaMatmulConfig,
    pub output_config: BlackboxOutputConfig,
}

impl TileAttentionConfig for BlackboxAcceleratedAttentionConfig {
    type ScoreMatmulConfig = CmmaMatmulConfig;
    type SoftmaxConfig = BlackboxSoftmaxConfig;
    type ValueMatmulConfig = CmmaMatmulConfig;
    type AttentionOutputConfig = BlackboxOutputConfig;

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

impl TileAttentionFamily for BlackboxAcceleratedTileAttention {
    type TileAttention<F: AttentionPrecision> = BlackboxAcceleratedTileAttention;

    type Config = BlackboxAcceleratedAttentionConfig;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let inner_layout = if blueprint.two_rows_in_array_tile {
            InnerLayout::SplitRows
        } else {
            InnerLayout::Contiguous
        };
        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let plane_dim = blueprint.plane_dim;
        let tile_size = blueprint.tiling_scheme.tile_size;
        validate(
            device_props,
            BlackboxAcceleratedAttentionConfig {
                shared: SharedTileAttentionConfig {
                    plane_dim,
                    num_planes,
                    attention_tile_size: tile_size,
                },
                inner_layout,
                score_matmul_config: CmmaMatmulConfig {
                    tile_size: blueprint
                        .tiling_scheme
                        .tile_size
                        .to_score_matmul_tile_size(),
                },
                softmax_config: BlackboxSoftmaxConfig {
                    tile_size,
                    plane_dim,
                    inner_layout,
                    causal_mask: blueprint.causal,
                    materialized_mask: blueprint.masked,
                    num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                },
                value_matmul_config: CmmaMatmulConfig {
                    tile_size: blueprint
                        .tiling_scheme
                        .tile_size
                        .to_value_matmul_tile_size(),
                },
                output_config: BlackboxOutputConfig {
                    tile_size,
                    num_planes,
                    plane_dim,
                    inner_layout,
                },
            },
            blueprint.vector_sizes.mask,
            dtypes,
        )
    }
}

fn validate(
    device_props: &DeviceProperties,
    config: BlackboxAcceleratedAttentionConfig,
    line_sizes_mask: VectorSize,
    dtypes: &AttentionElems,
) -> Result<BlackboxAcceleratedAttentionConfig, AttentionSetupError> {
    if dtypes.query_global != dtypes.query_tile {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Query global and tile types must be the same because no stage to cast in between",
        )));
    }

    if !device_props.features.cmma.contains(&MmaConfig {
        a_type: dtypes.query_tile,
        b_type: dtypes.key_value_tile,
        cd_type: dtypes.softmax_acc,
        m: config.tile_size().seq_q,
        k: config.tile_size().head_dim,
        n: config.tile_size().seq_kv,
    }) {
        return Err(AttentionSetupError::Unavailable(
            AttentionAvailabilityError::MatmulInstructionUnavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs: dtypes.query_tile,
                    rhs: dtypes.key_value_tile,
                    output: dtypes.softmax_acc,
                    size: Some(config.tile_size().to_score_matmul_tile_size()),
                },
            ),
        ));
    }
    if !device_props.features.cmma.contains(&MmaConfig {
        a_type: dtypes.softmax_lhs,
        b_type: dtypes.key_value_tile,
        cd_type: dtypes.accumulator,
        m: config.tile_size().seq_q,
        k: config.tile_size().seq_kv,
        n: config.tile_size().val_dim,
    }) {
        return Err(AttentionSetupError::Unavailable(
            AttentionAvailabilityError::MatmulInstructionUnavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs: dtypes.softmax_lhs,
                    rhs: dtypes.key_value_tile,
                    output: dtypes.accumulator,
                    size: Some(config.tile_size().to_value_matmul_tile_size()),
                },
            ),
        ));
    }

    if line_sizes_mask > 1 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Line size mask > 1 not supported yet on accelerated tile attention",
        )));
    }

    let softmax_num_rows = config.shared.attention_tile_size.seq_q;
    let softmax_num_cols = config.shared.attention_tile_size.seq_kv;
    let softmax_total = softmax_num_rows * softmax_num_cols;

    if !softmax_total.is_multiple_of(config.shared.plane_dim) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Softmax size should be divisible by plane dim",
        )));
    }

    if config.inner_layout == InnerLayout::Contiguous && softmax_num_rows > config.shared.plane_dim
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "More than one row per unit not supported with this inner layout",
        )));
    }

    if config.inner_layout == InnerLayout::SplitRows
        && !softmax_total.is_multiple_of(2 * config.shared.plane_dim)
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "With split rows, units must have two elements each",
        )));
    }

    if config.shared.attention_tile_size.head_dim < config.shared.attention_tile_size.val_dim {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Can't have tile head_dim < tile val dim (not sure why)",
        )));
    }

    Ok(config)
}
