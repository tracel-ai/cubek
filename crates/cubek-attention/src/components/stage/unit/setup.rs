use std::marker::PhantomData;

use crate::{
    components::{
        stage::{
            AttentionTilingLayout, PartitionAttentionConfig, SharedPartitionAttentionConfig,
            unit::{UnitPartitionAttention, UnitPartitionStageConfig},
            validate,
        },
        tile::TileAttentionFamily,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        attention_types::*,
    },
};
use cubecl::prelude::ReadWrite;
use cubek_matmul::{
    components::{
        CubeDimResource,
        stage::{StageFamily, StageMemoryConfig, SwizzleMode},
        tile::io::Strided,
    },
    definition::MatrixLayout,
};

use crate::components::stage::StageAttentionFamily;

pub struct UnitPartitionStageAttentionFamily<
    TA: TileAttentionFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(TA, SK, SV, SO)>,
}

impl<
    TA: TileAttentionFamily,
    SK: StageFamily<TileKind = Strided>,
    SV: StageFamily<TileKind = Strided>,
    SO: StageFamily<ReadWrite, TileKind = Strided>,
> StageAttentionFamily for UnitPartitionStageAttentionFamily<TA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = UnitPartitionAttention<
        AP,
        SK::Stage<KS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, AttentionTilingLayout>,
        TA::TileAttention<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig<TA::Config>;

    fn expand_config(
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let tile_config = TA::expand_config(blueprint)?;
        let compute_resources = match TA::computation_resources()? {
            CubeDimResource::Units(units) => {
                CubeDimResource::Units(units * blueprint.tiling_scheme.stage_size.seq_q)
            }
            _ => {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Error: Expected unit tile attention, got a plane tile attention".to_string(),
                )));
            }
        };
        let num_planes = compute_resources.num_planes(blueprint.plane_dim)?;

        let key_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.head_dim,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.head_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.key as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
            dtype: dtypes.key_stage,
        };

        let value_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.val_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.value as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
            dtype: dtypes.value_stage,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_q,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.val_dim,
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            // Each unit has its slot in row direction
            partitions_per_stage_along_row: num_planes * blueprint.plane_dim,
            partitions_per_stage_along_col: 1,
            line_size: blueprint.line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
            dtype: dtypes.out_stage,
        };

        validate(PartitionAttentionConfig::Unit(UnitPartitionStageConfig {
            shared: SharedPartitionAttentionConfig {
                tile_config,
                partition_size: blueprint.tiling_scheme.partition_size,
                stage_size: blueprint.tiling_scheme.stage_size,
                reuse_key_value: blueprint.reuse_key_value,
                num_planes,
                key_smem_config,
                value_smem_config,
                out_smem_config,
            },
        }))
    }
}
