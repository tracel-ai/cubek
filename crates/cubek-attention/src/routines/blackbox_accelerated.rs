use cubecl::CubeDim;
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::launch::AttentionTileSize;
use crate::launch::{
    AttentionBlueprint, AttentionDefinition, AttentionElems, AttentionPartitionSize,
    AttentionSetupError, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
    RoutineStrategy,
};
use crate::routines::{DeviceSettings, LaunchInfo};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedRoutine {}

impl Routine for BlackboxAcceleratedRoutine {
    type TileAttention = BlackboxAcceleratedTileAttention;
    type StageAttention = PlanePartitionStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    type Strategy = ();

    fn prepare(
        definition: &AttentionDefinition,
        device_settings: &DeviceSettings,
        strategy: RoutineStrategy<Self>,
    ) -> Result<super::LaunchInfo, AttentionSetupError> {
        let blueprint = blueprint(definition, device_settings, strategy)?;

        let dtypes = AttentionElems::from_global_types(
            &definition.global_dtypes,
            &definition.options.accumulator_precision,
        );

        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let cube_dim = CubeDim::new_2d(blueprint.plane_dim, num_planes);

        let cube_count_plan = blueprint
            .hypercube_blueprint
            .cube_count_plan(&definition.dims, &blueprint);

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
        })
    }
}

fn blueprint(
    definition: &AttentionDefinition,
    launch_settings: &DeviceSettings,
    strategy: RoutineStrategy<BlackboxAcceleratedRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        RoutineStrategy::Forced(attention_blueprint) => validate(definition, attention_blueprint),
        RoutineStrategy::Inferred(_) => {
            #[cfg(target_os = "macos")]
            let tile_size = AttentionTileSize {
                seq_q: 8,
                head_dim: 8,
                seq_kv: 8,
                val_dim: 8,
            };
            #[cfg(not(target_os = "macos"))]
            let tile_size = AttentionTileSize {
                seq_q: 16,
                head_dim: 16,
                seq_kv: 16,
                val_dim: 16,
            };

            let partition_head_dim = definition.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = partition_head_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: 1,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: 1 },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint {},
                plane_dim: launch_settings.plane_dim,
                reuse_key_value: false,
                two_rows_in_array_tile: false,
                line_sizes: launch_settings.line_sizes.clone(),
                masked: definition.masked,
                causal: definition.options.causal,
                tiling_scheme,
                check_bounds: tiling_scheme.check_bounds(&definition.dims),
            };

            validate(definition, blueprint)
        }
    }
}

fn validate(
    definition: &AttentionDefinition,
    blueprint: AttentionBlueprint,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    if definition.dims.head_dim as u32 % blueprint.tiling_scheme.tile_size.head_dim != 0 {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tile size head dim must divide problem head dim".to_string(),
        )));
    }

    if blueprint.tiling_scheme.partition_size.head_dim * blueprint.tiling_scheme.tile_size.head_dim
        != definition.dims.head_dim as u32
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tiling scheme's total head dim must equal problem's head dim".to_string(),
        )));
    }

    Ok(blueprint)
}
