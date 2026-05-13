//! StageMatmul instance enum + selector. Mirrors
//! [`crate::components::tile`]'s `TileMatmul` / `TileMatmulKind` pair.
//! Replaces the (`StageMatmul`, `StageMatmulFamily`, `StageConfig`) trait
//! triple with a single enum carrying the matmul instance directly — each
//! variant *is* the matmul (no wrapper config). The old
//! `SharedPartitionMatmulConfig` / `PartitionMatmulConfig` /
//! `UnitPartitionedStageConfig` / `PlanePartitionedStageConfig` wrappers and
//! the [`crate::components::stage::StageConfig`] trait are deleted in PR 6;
//! a temporary `impl StageConfig for StageMatmul` lives at the bottom of this
//! file to keep callers compiling in the meantime.

use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{
    CubeDimResource, InvalidConfigError, MatrixLayout, PartitionSize, StageSize,
    stage::StageMemoryConfig,
    tile::{PartitionBuffering, PartitionSchedulerScheme, TileMatmul},
};

use crate::components::global::{MatmulPlaneCounts, PlaneFlowConfig};
use crate::components::stage::{NumStages, StageConfig};
use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

/// Data carried by both [`StageMatmul`] variants. Today the unit- and
/// plane-partitioned flows hold the same fields (only the partition flavor —
/// which compute primitive owns each partition — differs); they share one
/// struct here and the variant selects between the two flavors.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedStageMatmul {
    pub tile_matmul: TileMatmul,
    pub partition_size: PartitionSize,
    pub partition_buffering: PartitionBuffering,
    pub plane_flow_config: PlaneFlowConfig,
    pub plane_dim: u32,
    pub stage_size: StageSize,
    pub partition_schedule_scheme: PartitionSchedulerScheme,
    pub lhs_smem_config: StageMemoryConfig,
    pub rhs_smem_config: StageMemoryConfig,
    pub acc_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

/// Stage-level matmul instance. The variant tags which compute primitive owns
/// each partition; the carried [`PartitionedStageMatmul`] is the rest of the
/// configuration. This is both the runtime selector and the comptime
/// configuration — pick the variant for the kernel you want, then forward the
/// value into the global / batch layers where its accessors drive allocation
/// and execution.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StageMatmul {
    UnitPartitioned(PartitionedStageMatmul),
    PlanePartitioned(PartitionedStageMatmul),
}

/// Selector for the stage-level matmul kind, used before per-kind config exists.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StageMatmulKind {
    UnitPartitioned,
    PlanePartitioned,
}

impl StageMatmul {
    /// Inner instance, shared between both variants.
    pub fn data(&self) -> &PartitionedStageMatmul {
        match self {
            StageMatmul::UnitPartitioned(m) | StageMatmul::PlanePartitioned(m) => m,
        }
    }

    pub fn kind(&self) -> StageMatmulKind {
        match self {
            StageMatmul::UnitPartitioned(_) => StageMatmulKind::UnitPartitioned,
            StageMatmul::PlanePartitioned(_) => StageMatmulKind::PlanePartitioned,
        }
    }
}

impl StageMatmulKind {
    /// Constructs the [`StageMatmul`] instance based on the matmul problem,
    /// selection, vector sizes, and number of stages.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_stage_matmul(
        &self,
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        plane_flow_config: PlaneFlowConfig,
        num_stages: NumStages,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<StageMatmul, MatmulSetupError> {
        let plane_counts = MatmulPlaneCounts::new(blueprint.load_flows, plane_flow_config.counts);

        let lhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.lhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.k,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.k as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.k as u32,
            vector_size: vector_sizes.lhs as u32,
            matrix_layout: blueprint.lhs_layout,
            swizzle: blueprint.swizzle_modes.lhs,
            num_stages: num_stages.lhs,
            dtype: dtypes.lhs_stage,
        };

        let rhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.rhs,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.k,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.k as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.k as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            vector_size: vector_sizes.rhs as u32,
            matrix_layout: blueprint.rhs_layout,
            swizzle: blueprint.swizzle_modes.rhs,
            num_stages: num_stages.rhs,
            dtype: dtypes.rhs_stage,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes: plane_counts.out,
            elements_per_tile_along_row: blueprint.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: blueprint.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: blueprint.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: blueprint.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: blueprint.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: blueprint.tiling_scheme.stage_size.n as u32,
            vector_size: vector_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: blueprint.swizzle_modes.out,
            num_stages: 1,
            dtype: dtypes.acc_stage,
        };

        let data = PartitionedStageMatmul {
            tile_matmul: blueprint.tile_matmul.expand_tile_matmul(
                device_props,
                blueprint,
                dtypes,
                vector_sizes,
            )?,
            partition_size: blueprint.tiling_scheme.partition_size,
            partition_buffering: blueprint.partition_buffering,
            plane_flow_config,
            plane_dim: blueprint.plane_dim,
            stage_size: blueprint.tiling_scheme.stage_size,
            partition_schedule_scheme: PartitionSchedulerScheme::Naive,
            lhs_smem_config,
            rhs_smem_config,
            acc_smem_config: out_smem_config,
            out_smem_config,
        };

        Ok(match self {
            StageMatmulKind::UnitPartitioned => StageMatmul::UnitPartitioned(data),
            StageMatmulKind::PlanePartitioned => StageMatmul::PlanePartitioned(data),
        })
    }

    /// Compute resources required for this stage matmul on the given blueprint.
    pub fn cubedim_resource(
        &self,
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        let inner = blueprint.tile_matmul.cubedim_resource()?;
        let factor = blueprint.tiling_scheme.partitions_per_stage_along_m()
            * blueprint.tiling_scheme.partitions_per_stage_along_n();
        match (self, inner) {
            (StageMatmulKind::PlanePartitioned, CubeDimResource::Planes(planes)) => {
                Ok(CubeDimResource::Planes(planes * factor))
            }
            (StageMatmulKind::PlanePartitioned, _) => Err(Box::new(
                "Error: Tried to use a plane stage matmul with a unit tile matmul.".to_string(),
            )),
            (StageMatmulKind::UnitPartitioned, CubeDimResource::Units(units)) => {
                Ok(CubeDimResource::Units(units * factor))
            }
            (StageMatmulKind::UnitPartitioned, _) => Err(Box::new(
                "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
            )),
        }
    }

    pub fn validate_blueprint<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        match self {
            StageMatmulKind::PlanePartitioned => {
                let num_planes_needed = blueprint.tiling_scheme.partitions_per_stage_along_m()
                    * blueprint.tiling_scheme.partitions_per_stage_along_n();
                let num_compute_planes =
                    self.cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;

                if num_compute_planes != num_planes_needed {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
                    ))));
                }
            }
            StageMatmulKind::UnitPartitioned => {
                let working_units = blueprint.tiling_scheme.partitions_per_stage_along_m()
                    * blueprint.tiling_scheme.partitions_per_stage_along_n();
                let num_compute_planes =
                    self.cubedim_resource(blueprint)?.num_planes(blueprint.plane_dim)?;
                let num_units = blueprint.plane_dim * num_compute_planes;

                if num_units != working_units {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Error: Number of units {num_units} should be {working_units}."
                    ))));
                }
            }
        }

        if blueprint.partition_buffering == PartitionBuffering::Double
            && blueprint.tiling_scheme.tiles_per_stage_partition_along_n() < 2
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            )));
        }

        blueprint
            .tile_matmul
            .validate_blueprint(client, blueprint, dtypes, vector_sizes)
    }
}

// =====================================================================
// Temporary StageConfig impl. Lets the existing trait-bound callers
// (e.g. `MP: ... where Self::Config: StageConfig`) keep compiling while
// PR 6 cuts them over and deletes the trait stack.
// =====================================================================

impl StageConfig for StageMatmul {
    fn elements_in_stage_m(&self) -> u32 {
        let d = self.data();
        d.stage_size.m() * d.partition_size.m() * d.tile_matmul.elements_in_tile_m()
    }

    fn elements_in_stage_n(&self) -> u32 {
        let d = self.data();
        d.stage_size.n() * d.partition_size.n() * d.tile_matmul.elements_in_tile_n()
    }

    fn elements_in_stage_k(&self) -> u32 {
        let d = self.data();
        d.stage_size.k() * d.partition_size.k() * d.tile_matmul.elements_in_tile_k()
    }

    fn elements_in_tile_k(&self) -> u32 {
        self.data().tile_matmul.elements_in_tile_k()
    }

    fn tiles_in_partition_mn(&self) -> u32 {
        let p = self.data().partition_size;
        p.m() * p.n()
    }

    fn num_main_flow_planes(&self) -> u32 {
        self.data().plane_flow_config.main_flow_count()
    }

    fn plane_dim(&self) -> u32 {
        self.data().plane_dim
    }

    fn plane_flow_config(&self) -> PlaneFlowConfig {
        self.data().plane_flow_config
    }

    fn lhs_smem_config(&self) -> StageMemoryConfig {
        self.data().lhs_smem_config
    }

    fn rhs_smem_config(&self) -> StageMemoryConfig {
        self.data().rhs_smem_config
    }

    fn acc_smem_config(&self) -> StageMemoryConfig {
        self.data().acc_smem_config
    }

    fn out_smem_config(&self) -> StageMemoryConfig {
        self.data().out_smem_config
    }
}
