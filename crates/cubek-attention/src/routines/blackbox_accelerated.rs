use cubecl::{
    client::ComputeClient,
    prelude::CubePrimitive,
    {CubeDim, Runtime},
};
use cubek_matmul::{
    components::{global::PartitionedStageFamily, stage::StridedStageFamily},
    routines::find_instruction_size,
};

use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use crate::{
    components::stage::plane::PlanePartitionStageAttentionFamily,
    components::tile::attention::blackbox::BlackboxAcceleratedTileAttention,
    definition::AttentionTileSize,
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
            let is_supported = |client: &ComputeClient<R>, mma| {
                client.properties().features.matmul.cmma.contains(&mma)
            };

            let supported_sizes = |client: &ComputeClient<R>, lhs_ty, rhs_ty, acc_ty| {
                client
                    .properties()
                    .features
                    .matmul
                    .cmma
                    .iter()
                    .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
                    .map(|it| (it.m, it.n, it.k).into())
                    .collect::<Vec<_>>()
            };
            let map_err = |err| {
                AttentionSetupError::Unavailable(
                    crate::definition::AttentionAvailabilityError::MatmulInstructionUnavailable(
                        err,
                    ),
                )
            };

            let tile_size_score_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (dtypes.query_tile, dtypes.key_value_tile, dtypes.softmax_acc),
                (
                    problem.dims.seq_q,
                    problem.dims.seq_kv,
                    problem.dims.head_dim,
                )
                    .into(),
                (None, None, None),
                is_supported,
                supported_sizes,
            )
            .map_err(map_err)?;

            let values_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (
                    dtypes.softmax_lhs,
                    dtypes.key_value_tile,
                    dtypes.accumulator,
                ),
                (
                    problem.dims.seq_q,
                    problem.dims.val_dim,
                    problem.dims.seq_kv,
                )
                    .into(),
                (
                    Some(tile_size_score_matmul.m),
                    None,
                    Some(tile_size_score_matmul.n),
                ),
                is_supported,
                supported_sizes,
            )
            .map_err(map_err)?;

            if tile_size_score_matmul.m != values_matmul.m {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Seq_q mismatch: `m` of score_matmul does not match `m` of values_matmul. ",
                )));
            }

            if tile_size_score_matmul.n != values_matmul.k {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Seq_kv mismatch: `n` of score_matmul does not match `k` of values_matmul. ",
                )));
            }

            let tile_size = AttentionTileSize {
                seq_q: tile_size_score_matmul.m,
                head_dim: tile_size_score_matmul.k,
                seq_kv: tile_size_score_matmul.n,
                val_dim: values_matmul.n,
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: strategy.seq_q as u32,
                    head_dim: partition_head_dim,
                    seq_kv: strategy.seq_kv as u32,
                    val_dim: partition_val_dim,
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
    if !(problem.dims.seq_q as u32)
        .is_multiple_of(blueprint.tiling_scheme.elements_in_stage_seq_q())
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Stage seq_q must divide problem seq_q".to_string(),
        )));
    }

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
