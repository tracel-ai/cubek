use cubecl::{
    prelude::CubePrimitive as _,
    {CubeDim, Runtime},
};
use cubek_matmul::{
    components::CubeDimResource,
    components::{global::PartitionedStageFamily, stage::StridedStageFamily},
};

use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
    HypercubeBlueprint,
};
use crate::{
    components::stage::unit::UnitPartitionStageAttentionFamily,
    components::tile::TileAttentionFamily, components::tile::attention::unit::UnitTileAttention,
};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};
use crate::{
    launch::BlueprintStrategy,
    routines::{DeviceSettings, LaunchInfo},
};

#[derive(Debug, Clone)]
pub struct UnitRoutine {}

impl Routine for UnitRoutine {
    type TileAttention = UnitTileAttention;
    type StageAttention = UnitPartitionStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    type Strategy = ();
    type Blueprint = AttentionBlueprint;

    fn prepare<R: Runtime>(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings<R>,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let blueprint = blueprint(problem, device_settings, strategy)?;

        let dtypes = AttentionElems::from_global_types(
            &problem.global_dtypes,
            half::f16::as_type_native_unchecked().storage_type(),
            &problem.options.accumulator_precision,
        );

        let compute_resources = match Self::TileAttention::computation_resources()? {
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
        let cube_dim = CubeDim::new_2d(blueprint.plane_dim, num_planes);
        let cube_count_plan =
            blueprint.cube_count_plan(&problem.dims, &device_settings.max_cube_count);

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
            address_type: problem.address_type,
        })
    }
}

fn blueprint<R: Runtime>(
    problem: &AttentionProblem,
    launch_settings: &DeviceSettings<R>,
    strategy: BlueprintStrategy<UnitRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(_) => {
            let tile_size = AttentionTileSize::from_max_vector_sizes(&launch_settings.vector_sizes);

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            let plane_dim = launch_settings.plane_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: 1,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: plane_dim },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint::builder().build(),
                tiling_scheme,
                plane_dim,
                two_rows_in_array_tile: false,
                vector_sizes: launch_settings.vector_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
                check_bounds: tiling_scheme.check_bounds(&problem.dims),
            };

            validate(problem, blueprint)
        }
    }
}

fn validate(
    problem: &AttentionProblem,
    blueprint: AttentionBlueprint,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    if !(problem.dims.head_dim as u32).is_multiple_of(blueprint.tiling_scheme.tile_size.head_dim) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tile size head dim must divide problem head dim".to_string(),
        )));
    }

    if blueprint.tiling_scheme.partition_size.head_dim * blueprint.tiling_scheme.tile_size.head_dim
        != problem.dims.head_dim as u32
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "Tiling scheme's total head dim ({}) does not match problem's head dim ({})",
            blueprint.tiling_scheme.partition_size.head_dim
                * blueprint.tiling_scheme.tile_size.head_dim,
            problem.dims.head_dim
        ))));
    }

    Ok(blueprint)
}
