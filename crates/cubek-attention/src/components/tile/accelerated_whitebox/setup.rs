use cubecl::ir::DeviceProperties;
use cubecl::ir::LineSize;
use cubek_matmul::components::CubeDimResource;
use cubek_std::tile::mma::MmaIOConfig;

use crate::components::tile::SharedTileAttentionConfig;
use crate::components::tile::TileAttentionConfig;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::accelerated_whitebox::WhiteboxAcceleratedTileAttention;
use crate::definition::AttentionBlueprint;
use crate::definition::AttentionElems;
use crate::definition::AttentionPrecision;
use crate::definition::AttentionSetupError;
use crate::definition::AttentionTileSize;
use crate::definition::InvalidConfigError;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct WhiteboxAcceleratedAttentionMatmulConfig {
    pub shared: SharedTileAttentionConfig,
    pub out_smem_line_size: usize,
    pub score_mma_io_config: MmaIOConfig,
    pub value_mma_io_config: MmaIOConfig,
}

impl TileAttentionConfig for WhiteboxAcceleratedAttentionMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.shared.plane_dim
    }

    fn num_planes(&self) -> u32 {
        self.shared.num_planes
    }

    fn attention_tile_size(&self) -> AttentionTileSize {
        self.shared.attention_tile_size
    }

    fn num_rows_per_unit(&self) -> u32 {
        2
    }

    fn causal_mask(&self) -> bool {
        self.shared.causal_mask
    }

    fn materialized_mask(&self) -> bool {
        self.shared.materialized_mask
    }
}

impl TileAttentionFamily for WhiteboxAcceleratedTileAttention {
    type TileAttention<F: AttentionPrecision> = WhiteboxAcceleratedTileAttention;

    type Config = WhiteboxAcceleratedAttentionMatmulConfig;

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
        validate(
            device_props,
            WhiteboxAcceleratedAttentionMatmulConfig {
                shared: SharedTileAttentionConfig {
                    plane_dim: blueprint.plane_dim,
                    num_planes: blueprint.tiling_scheme.stage_size.seq_q,
                    attention_tile_size: blueprint.tiling_scheme.tile_size,
                    causal_mask: blueprint.causal,
                    materialized_mask: blueprint.masked,
                },
                out_smem_line_size: blueprint.line_sizes.out,
                score_mma_io_config: MmaIOConfig::new(
                    device_props,
                    dtypes.query_global,
                    dtypes.key_stage,
                    dtypes.softmax_acc,
                ),
                value_mma_io_config: MmaIOConfig::new(
                    device_props,
                    dtypes.softmax_lhs,
                    dtypes.value_stage,
                    dtypes.accumulator,
                ),
            },
            blueprint.reuse_key_value,
            blueprint.line_sizes.mask,
            dtypes,
        )
    }
}

fn validate(
    _device_props: &DeviceProperties,
    config: WhiteboxAcceleratedAttentionMatmulConfig,
    _reuse_key_value: bool,
    _line_sizes_mask: LineSize,
    dtypes: &AttentionElems,
) -> Result<WhiteboxAcceleratedAttentionMatmulConfig, AttentionSetupError> {
    if dtypes.query_global != dtypes.query_tile {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Query global and tile types must be the same because no stage to cast in between",
        )));
    }

    // todo!();
    // if !device_props.features.cmma.contains(&MmaConfig {
    //     a_type: dtypes.query_tile,
    //     b_type: dtypes.key_value_tile,
    //     cd_type: dtypes.softmax_acc,
    //     m: config.attention_tile_size().seq_q,
    //     k: config.attention_tile_size().head_dim,
    //     n: config.attention_tile_size().seq_kv,
    // }) {
    //     return Err(AttentionSetupError::Unavailable(
    //         AttentionAvailabilityError::CmmaInstructionUnavailable {
    //             lhs: dtypes.query_tile,
    //             rhs: dtypes.key_value_tile,
    //             output: dtypes.softmax_acc,
    //         },
    //     ));
    // }
    // if !device_props.features.cmma.contains(&MmaConfig {
    //     a_type: dtypes.softmax_lhs,
    //     b_type: dtypes.key_value_tile,
    //     cd_type: dtypes.accumulator,
    //     m: config.attention_tile_size().seq_q,
    //     k: config.attention_tile_size().seq_kv,
    //     n: config.attention_tile_size().val_dim,
    // }) {
    //     return Err(AttentionSetupError::Unavailable(
    //         AttentionAvailabilityError::CmmaInstructionUnavailable {
    //             lhs: dtypes.softmax_acc,
    //             rhs: dtypes.key_value_tile,
    //             output: dtypes.accumulator,
    //         },
    //     ));
    // }

    // todo!();
    // if line_sizes_mask > 1 {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "Line size mask > 1 not supported yet on accelerated tile attention",
    //     )));
    // }

    // let softmax_num_rows = config.shared.attention_tile_size.seq_q;
    // let softmax_num_cols = config.shared.attention_tile_size.seq_kv;
    // let softmax_total = softmax_num_rows * softmax_num_cols;

    // if softmax_total % config.shared.plane_dim != 0 {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "Softmax size should be divisible by plane dim",
    //     )));
    // }

    // if config.inner_layout == InnerLayout::Contiguous && softmax_num_rows > config.shared.plane_dim
    // {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "More than one row per unit not supported with this inner layout",
    //     )));
    // }

    // if config.inner_layout == InnerLayout::SplitRows
    //     && softmax_total % (2 * config.shared.plane_dim) != 0
    // {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "With split rows, units must have two elements each",
    //     )));
    // }

    // if config.shared.attention_tile_size.head_dim < config.shared.attention_tile_size.val_dim {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "Can't have tile head_dim < tile val dim (not sure why)",
    //     )));
    // }

    // if reuse_key_value {
    //     return Err(AttentionSetupError::InvalidConfig(Box::new(
    //         "Can't reuse key/value because unimplemented",
    //     )));
    // }

    Ok(config)
}
