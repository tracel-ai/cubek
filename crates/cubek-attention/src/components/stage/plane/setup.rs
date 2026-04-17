use std::marker::PhantomData;

use crate::{
    components::{
        stage::{
            AttentionTilingLayout, PartitionAttentionConfig, SharedPartitionAttentionConfig,
            plane::{PlanePartitionAttention, PlanePartitionStageConfig},
        },
        tile::TileAttentionFamily,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        attention_types::*,
    },
};
use cubecl::{ir::DeviceProperties, prelude::ReadWrite};
use cubek_matmul::components::stage::StageFamily;
use cubek_std::{
    MatrixLayout,
    stage::{StageMemoryConfig, SwizzleMode},
};

use crate::components::stage::StageAttentionFamily;

pub struct PlanePartitionStageAttentionFamily<
    TA: TileAttentionFamily,
    SK: StageFamily,
    SV: StageFamily,
    SO: StageFamily<ReadWrite>,
> {
    _phantom: PhantomData<(TA, SK, SV, SO)>,
}

impl<TA: TileAttentionFamily, SK: StageFamily, SV: StageFamily, SO: StageFamily<ReadWrite>>
    StageAttentionFamily for PlanePartitionStageAttentionFamily<TA, SK, SV, SO>
{
    type Attention<AP: AttentionPrecision> = PlanePartitionAttention<
        AP,
        SK::Stage<KS<AP>, KSS<AP>, AttentionTilingLayout>,
        SV::Stage<VS<AP>, VSS<AP>, AttentionTilingLayout>,
        SO::Stage<OS<AP>, OSS<AP>, AttentionTilingLayout>,
        TA::TileAttention<AP>,
    >;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig<TA::Config>;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let num_planes = blueprint.tiling_scheme.stage_size.seq_q
            * TA::computation_resources()?.num_planes(blueprint.plane_dim)?;

        let tile_config = TA::expand_config(device_props, blueprint, dtypes)?;

        let key_smem_config = StageMemoryConfig {
            num_planes,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.seq_kv,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.head_dim,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.seq_kv,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.head_dim,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            vector_size: blueprint.vector_sizes.key as u32,
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
            vector_size: blueprint.vector_sizes.value as u32,
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
            // Each plane has its slot in row direction
            partitions_per_stage_along_row: num_planes,
            partitions_per_stage_along_col: 1,
            vector_size: blueprint.vector_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
            dtype: dtypes.out_stage,
        };

        Ok(PartitionAttentionConfig::Plane(PlanePartitionStageConfig {
            shared: SharedPartitionAttentionConfig {
                partition_size: blueprint.tiling_scheme.partition_size,
                stage_size: blueprint.tiling_scheme.stage_size,
                num_planes,
                key_smem_config,
                value_smem_config,
                out_smem_config,
                tile_config,
            },
        }))
    }
}
