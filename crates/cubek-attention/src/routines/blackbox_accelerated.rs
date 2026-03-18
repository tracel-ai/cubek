use cubecl::prelude::CubePrimitive;
use cubecl::{CubeDim, Runtime};
use cubek_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};
use cubek_matmul::routines::find_instruction_size;

use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::attention::blackbox::BlackboxAcceleratedTileAttention;
use crate::definition::AttentionTileSize;
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use crate::launch::BlueprintStrategy;
use crate::routines::{DeviceSettings, LaunchInfo};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedRoutine {}

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedStrategy {
    pub num_planes: u8,
    pub seq_q: u8,
    pub seq_kv: u8,
}

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

    type Strategy = BlackboxAcceleratedStrategy;
    type Blueprint = AttentionBlueprint;

    fn prepare<R: Runtime>(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings<R>,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let dtypes = AttentionElems::from_global_types(
            &problem.global_dtypes,
            half::f16::as_type_native_unchecked().storage_type(),
            &problem.options.accumulator_precision,
        );

        let blueprint = blueprint(problem, device_settings, &dtypes, strategy)?;

        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let cube_dim = CubeDim::new_2d(blueprint.plane_dim, num_planes);

        let cube_count_plan = blueprint
            .hypercube_blueprint
            .cube_count_plan(&problem.dims, &blueprint);

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
    device: &DeviceSettings<R>,
    dtypes: &AttentionElems,
    strategy: BlueprintStrategy<BlackboxAcceleratedRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => validate(problem, attention_blueprint),
        BlueprintStrategy::Inferred(strategy) => {
            let tile_size_score_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (dtypes.query_tile, dtypes.key_value_tile, dtypes.softmax_acc),
                problem.dims.seq_q,
                problem.dims.seq_kv,
                |client, mma| client.properties().features.cmma.contains(&mma),
                // TODO: Implement fallback
                |_, _, _, _| Vec::new(),
            )
            .map_err(|err| {
                AttentionSetupError::Unavailable(
                    crate::definition::AttentionAvailabilityError::MatmulInstructionUnavailable(
                        err,
                    ),
                )
            })?;

            let values_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (
                    dtypes.softmax_lhs,
                    dtypes.key_value_tile,
                    dtypes.accumulator,
                ),
                tile_size_score_matmul.m as usize,
                problem.dims.val_dim,
                |client, mma| client.properties().features.cmma.contains(&mma),
                // TODO: Implement fallback
                |_, _, _, _| Vec::new(),
            )
            .map_err(|err| {
                AttentionSetupError::Unavailable(
                    crate::definition::AttentionAvailabilityError::MatmulInstructionUnavailable(
                        err,
                    ),
                )
            })?;

            if tile_size_score_matmul.m != values_matmul.m {
                return Err(AttentionSetupError::InvalidConfig(Box::new("")));
            }

            if tile_size_score_matmul.n != values_matmul.k {
                return Err(AttentionSetupError::InvalidConfig(Box::new("")));
            }

            let tile_size = AttentionTileSize {
                seq_q: tile_size_score_matmul.m,
                head_dim: tile_size_score_matmul.k,
                seq_kv: tile_size_score_matmul.n,
                val_dim: values_matmul.n,
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: strategy.seq_q as u32,
                    head_dim: partition_head_dim,
                    seq_kv: strategy.seq_kv as u32,
                    val_dim: partition_head_dim,
                },
                stage_size: AttentionStageSize {
                    seq_q: strategy.num_planes as u32,
                },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint {},
                plane_dim: device.plane_dim,
                two_rows_in_array_tile: false,
                vector_sizes: device.vector_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
                tiling_scheme,
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
