use std::marker::PhantomData;

use crate::{
    definition::{
        AccumulatorOperand, MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError,
        MatmulVectorSizes,
    },
    multi_level::{
        args::{ConfigRuntimeArg, RuntimeConfig, *},
        components::{
            CubeDimResource,
            batch::{
                BatchMatmulFamily,
                partitioned_matmul::{
                    config::PartitionedBatchConfig,
                    matmul::{PartitionedBatchMatmul, matmul_entry},
                    partition::GlobalPartitionMatmul,
                },
            },
            global::{GlobalConfig, GlobalMatmulFamily},
            stage::NumStages,
        },
        definition::{BatchMatmulBlueprint, CubeMappingLaunch, MatmulTypes},
        stage::StageMemoryConfig,
    },
};
use cubecl::{ir::DeviceProperties, prelude::*};

/// Simple partitioned batch matmul family for any precision
pub struct PartitionedBatchMatmulFamily<
    RC: RuntimeConfig,
    GMM: GlobalMatmulFamily<RC>,
    S: GlobalPartitionMatmul,
> {
    _rc: PhantomData<RC>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

impl<RC: RuntimeConfig, GMM: GlobalMatmulFamily<RC>, S: GlobalPartitionMatmul> BatchMatmulFamily<RC>
    for PartitionedBatchMatmulFamily<RC, GMM, S>
{
    type Matmul<MP: MatmulTypes> = PartitionedBatchMatmul<RC, MP, GMM::Matmul<MP>, S>;
    type Config = PartitionedBatchConfig<GMM::Config>;
    type Blueprint = BatchMatmulBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::expand_config(device_props, blueprint, dtypes, vector_sizes)?;

        Ok(PartitionedBatchConfig::new(
            global_config,
            blueprint.tiling_scheme.global_partition_size,
        ))
    }

    fn num_stages() -> NumStages {
        GMM::num_stages()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = RC>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        config: ConfigRuntimeArg<MA>,
        cube_count_input: CubeMappingLaunch,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, GMM, S>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                config,
                cube_count_input,
                blueprint,
                dtypes.clone(),
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        GMM::cubedim_resource(blueprint, dtypes, vector_sizes)
    }

    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        // Multi-stage k-loops round their stage count up to `stage_buffering`, so a
        // blueprint that skips k-bounds checks must divide k by the whole group.
        let k_group = blueprint.tiling_scheme.elements_per_stage_along_k()
            * GMM::num_stages().stage_buffering();
        if !blueprint.check_k_bounds && !(problem.k as u32).is_multiple_of(k_group) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k-bounds checks are disabled but k={} is not a multiple of the k-loop group {k_group}",
                problem.k
            ))));
        }

        GMM::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)?;

        let global_config =
            GMM::expand_config(client.properties(), blueprint, dtypes, vector_sizes)?;
        let stage_config = global_config.stage_config();

        // Validate that the kernel's shared-memory footprint fits in the
        // per-cube budget the runtime reports.
        let acc = match problem.accumulator {
            AccumulatorOperand::Absent => None,
            AccumulatorOperand::Present => Some(global_config.acc_reader_config().smem_config),
        };
        let requested = requested_smem_bytes(
            &stage_config.lhs_smem_config(),
            &stage_config.rhs_smem_config(),
            &stage_config.out_smem_config(),
            acc.as_ref(),
        );
        let available = client.properties().hardware.max_shared_memory_size;

        if requested > available {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::SharedMemoryTooBig {
                    requested,
                    available,
                },
            ));
        }

        Ok(())
    }
}

/// Shared memory one cube of this matmul allocates.
///
/// The operands are not the whole footprint: the kernel also stages its
/// output. `expand_config` builds that stage once and hands it back as both
/// `acc_smem_config` and `out_smem_config`, so `out` covers both and is
/// counted a single time: at the writer's allocation, not the full stage
/// (see [`out_smem_bytes`]).
///
/// This has to match what the kernel allocates, in both directions. A total
/// that comes in under the real footprint admits a blueprint that then
/// over-requests at launch, which autotune profiling surfaces as a lost device
/// rather than a skipped candidate. A total that comes in over it rejects
/// blueprints that do fit, and with them the kernels autotune would have
/// picked.
fn requested_smem_bytes(
    lhs: &StageMemoryConfig,
    rhs: &StageMemoryConfig,
    out: &StageMemoryConfig,
    acc: Option<&StageMemoryConfig>,
) -> usize {
    let acc_bytes = acc.map(smem_bytes).unwrap_or(0);
    smem_bytes(lhs) + smem_bytes(rhs) + out_smem_bytes(out) + acc_bytes
}

/// Shared memory the writer allocates for the accumulator/output stage.
///
/// No writer holds the full stage: `PartitionedStage::new` clamps
/// `tiles_per_partition_along_{row,col}` to 1, so a plane (or unit) stages one
/// tile at a time and loops over its partition. This mirrors that clamp.
/// Charging the full stage instead rejected blueprints that fit: on Apple
/// silicon the simple cmma multi-rows kernel allocates 13312 bytes but was
/// charged 45056, past the 32768-byte budget, so the fastest kernel for the
/// shape became unavailable.
///
/// A launch that reads an accumulator operand stages it on top of this, in the
/// reader's own stage rather than the writer's clamped one; `requested_smem_bytes`
/// charges that separately, and only when the problem says one is present.
fn out_smem_bytes(out: &StageMemoryConfig) -> usize {
    let per_tile = StageMemoryConfig {
        tiles_per_partition_along_row: 1,
        tiles_per_partition_along_col: 1,
        ..*out
    };
    smem_bytes(&per_tile)
}

fn smem_bytes(cfg: &StageMemoryConfig) -> usize {
    cfg.elements_per_stage() as usize * cfg.num_stages as usize * cfg.dtype.size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_level::stage::SwizzleMode;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::MatrixLayout;

    /// A stage holding `rows * cols` f16 elements per stage, `num_stages` deep,
    /// so its footprint is `rows * cols * num_stages * 2` bytes.
    fn stage(rows: u32, cols: u32, num_stages: u32) -> StageMemoryConfig {
        StageMemoryConfig {
            num_planes: 1,
            elements_per_tile_along_row: rows,
            elements_per_tile_along_col: cols,
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            partitions_per_stage_along_row: 1,
            partitions_per_stage_along_col: 1,
            vector_size: 1,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages,
            dtype: ElemType::Float(FloatKind::F16),
        }
    }

    /// The operand stages are only part of what a cube allocates. These are the
    /// sizes from a blueprint that lost a device on a 48 KB budget: the operands
    /// land on exactly that budget, which a `>` check admits by equality, and
    /// the accumulator/output stage on top is what took the launch to 81920
    /// bytes. It has to be in the total the check sees.
    #[test]
    fn requested_bytes_include_the_output_stage() {
        let lhs = stage(128, 64, 2);
        let rhs = stage(64, 64, 2);
        let out = stage(128, 128, 1);

        assert_eq!(smem_bytes(&lhs), 32_768);
        assert_eq!(smem_bytes(&rhs), 16_384);
        assert_eq!(smem_bytes(&out), 32_768);

        assert_eq!(smem_bytes(&lhs) + smem_bytes(&rhs), 49_152);
        assert_eq!(requested_smem_bytes(&lhs, &rhs, &out, None), 81_920);
    }

    /// The accumulator and output are one buffer, so a config that reports the
    /// same stage for both must not be charged for it twice.
    #[test]
    fn the_output_stage_is_counted_once() {
        let lhs = stage(64, 64, 1);
        let rhs = stage(64, 64, 1);
        let out = stage(64, 64, 1);

        assert_eq!(requested_smem_bytes(&lhs, &rhs, &out, None), 3 * 8_192);
    }

    /// A launch that reads an accumulator operand allocates a reader stage the
    /// writer's clamp does not cover: the reader fills every tile before the
    /// k-loop, so it holds the whole stage, not one tile per partition. These
    /// are the stages of the f32 bias launch on Apple silicon, tile (4, 4, 4),
    /// partition (2, 2, 2), stage (16, 16, 1): the check saw 24576 bytes and
    /// the launch requested 90112, past the 32768-byte budget, because the
    /// 65536-byte reader stage was not in the total.
    #[test]
    fn requested_bytes_include_the_accumulator_reader_stage() {
        let lhs = tiled_stage((2, 1), (8, 1));
        let rhs = tiled_stage((2, 1), (8, 1));
        let out = tiled_stage((2, 2), (8, 8));

        assert_eq!(smem_bytes(&lhs), 4_096);
        assert_eq!(smem_bytes(&out), 65_536);
        assert_eq!(out_smem_bytes(&out), 16_384);

        assert_eq!(requested_smem_bytes(&lhs, &rhs, &out, None), 24_576);
        assert_eq!(requested_smem_bytes(&lhs, &rhs, &out, Some(&out)), 90_112);
    }

    /// A stage of 8x8 f32 tiles with the given partition structure, as the
    /// plane-partitioned blueprints build them.
    fn tiled_stage(
        tiles_per_partition: (u32, u32),
        partitions_per_stage: (u32, u32),
    ) -> StageMemoryConfig {
        StageMemoryConfig {
            num_planes: 4,
            elements_per_tile_along_row: 8,
            elements_per_tile_along_col: 8,
            tiles_per_partition_along_row: tiles_per_partition.0,
            tiles_per_partition_along_col: tiles_per_partition.1,
            partitions_per_stage_along_row: partitions_per_stage.0,
            partitions_per_stage_along_col: partitions_per_stage.1,
            vector_size: 1,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: SwizzleMode::None,
            num_stages: 1,
            dtype: ElemType::Float(FloatKind::F32),
        }
    }

    /// The writer never allocates the full out stage: `PartitionedStage::new`
    /// clamps `tiles_per_partition` to one, so each plane stages a single tile
    /// and loops. These are the stages of the simple cmma multi-rows blueprint
    /// on Apple silicon: tile (8, 8, 8), partition (4, 8, 2), stage (4, 1, 1),
    /// f32. The kernel allocates 13312 bytes; charging the full out stage says
    /// 45056, past the 32768-byte budget, and rejects the fastest kernel the
    /// device has for the shape.
    #[test]
    fn the_out_stage_is_charged_at_the_writers_allocation() {
        let lhs = tiled_stage((4, 2), (4, 1));
        let rhs = tiled_stage((2, 8), (1, 1));
        let out = tiled_stage((4, 8), (4, 1));

        assert_eq!(smem_bytes(&lhs), 8_192);
        assert_eq!(smem_bytes(&rhs), 4_096);
        assert_eq!(smem_bytes(&out), 32_768);
        assert_eq!(out_smem_bytes(&out), 1_024);

        assert_eq!(requested_smem_bytes(&lhs, &rhs, &out, None), 13_312);
    }
}
