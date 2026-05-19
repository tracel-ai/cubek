//! Stage-level matmul: the [`StageMatmul`] / [`StageMatmulKind`] enums that
//! dispatch over per-variant logic.
//!
//! Each stage-kind owns its validator, dispatcher, and `build` constructor in
//! its own module and implements [`StageVariant`]. This file holds *only*
//! dispatch: the enums themselves, accessor methods that read fields of the
//! carried data, and per-method 2-arm matches that forward into the variant's
//! [`StageVariant`] impl. Shared data (`PartitionedStageMatmul`, `NumStages`,
//! `StagePartitioner`, `init_*`) lives in [`common`].
//!
//! The generic `Stage` / `StageFamily` / `LoadStageFamily` traits and the
//! `StridedStageMemory` impls live in [`cubek_std::stage`]; cubek-matmul only
//! adds matmul-domain glue here.

#![allow(clippy::type_complexity)]

mod common;
mod plane_partitioned;
mod unit_partitioned;
mod variant;

pub use common::{
    NumStages, PartitionBuffering, PartitionedStageMatmul, Partitioner, PlanePartitioner,
    StagePartitioner, UnitPartitioner, init_a_fragment, init_accumulator, init_b_fragments,
};
pub use plane_partitioned::PlanePartitioned;
pub use unit_partitioned::UnitPartitioned;
pub use variant::StageVariant;

use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{
    CubeDimResource, InvalidConfigError, MatrixLayout, stage::StageMemoryConfig,
    tile::PartitionSchedulerScheme,
};

// Re-exports so callers can keep `use crate::components::stage::{Stage, …}`
// working without reaching into cubek_std directly.
pub use cubek_std::partition_coordinates;
pub use cubek_std::stage::{
    LoadStageFamily, Stage, StageFamily, StridedStageFamily, StridedStageMemory,
};

use crate::components::global::{MatmulPlaneCounts, PlaneFlowConfig};
use crate::definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint};

// =====================================================================
// StageMatmul — matmul-domain configuration enum
// =====================================================================

/// Stage-level matmul instance. The variant tags which compute primitive owns
/// each partition; the carried [`PartitionedStageMatmul`] is the rest of the
/// configuration.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StageMatmul {
    UnitPartitioned(PartitionedStageMatmul),
    PlanePartitioned(PartitionedStageMatmul),
}

/// Selector for the stage-level matmul kind, used before per-kind config exists.
///
/// All methods on this enum are pure 2-arm dispatchers that forward to the
/// matching variant's [`StageVariant`] impl.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StageMatmulKind {
    UnitPartitioned,
    PlanePartitioned,
}

impl StageMatmul {
    /// Inner instance, shared between both variants. Returned by reference.
    pub fn data(&self) -> &PartitionedStageMatmul {
        match self {
            StageMatmul::UnitPartitioned(m) | StageMatmul::PlanePartitioned(m) => m,
        }
    }

    /// Inner instance by value (since `PartitionedStageMatmul: Copy`).
    pub fn shared(&self) -> PartitionedStageMatmul {
        *self.data()
    }

    pub fn kind(&self) -> StageMatmulKind {
        match self {
            StageMatmul::UnitPartitioned(_) => StageMatmulKind::UnitPartitioned,
            StageMatmul::PlanePartitioned(_) => StageMatmulKind::PlanePartitioned,
        }
    }

    pub fn elements_in_stage_m(&self) -> u32 {
        let d = self.data();
        d.stage_size.m() * d.partition_size.m() * d.tile_matmul.elements_in_tile_m()
    }

    pub fn elements_in_stage_n(&self) -> u32 {
        let d = self.data();
        d.stage_size.n() * d.partition_size.n() * d.tile_matmul.elements_in_tile_n()
    }

    pub fn elements_in_stage_k(&self) -> u32 {
        let d = self.data();
        d.stage_size.k() * d.partition_size.k() * d.tile_matmul.elements_in_tile_k()
    }

    pub fn elements_in_tile_k(&self) -> u32 {
        self.data().tile_matmul.elements_in_tile_k()
    }

    pub fn tiles_in_partition_mn(&self) -> u32 {
        let p = self.data().partition_size;
        p.m() * p.n()
    }

    pub fn num_main_flow_planes(&self) -> u32 {
        self.data().plane_flow_config.main_flow_count()
    }

    pub fn lhs_smem_config(&self) -> StageMemoryConfig {
        self.data().lhs_smem_config
    }

    pub fn rhs_smem_config(&self) -> StageMemoryConfig {
        self.data().rhs_smem_config
    }

    pub fn acc_smem_config(&self) -> StageMemoryConfig {
        self.data().acc_smem_config
    }

    pub fn out_smem_config(&self) -> StageMemoryConfig {
        self.data().out_smem_config
    }

    pub fn plane_dim(&self) -> u32 {
        self.data().plane_dim
    }

    pub fn plane_flow_config(&self) -> PlaneFlowConfig {
        self.data().plane_flow_config
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
            StageMatmulKind::UnitPartitioned => UnitPartitioned::build(data),
            StageMatmulKind::PlanePartitioned => PlanePartitioned::build(data),
        })
    }

    /// Compute resources required for this stage matmul on the given blueprint.
    pub fn cubedim_resource(
        &self,
        blueprint: &TilingBlueprint,
    ) -> Result<CubeDimResource, InvalidConfigError> {
        match self {
            StageMatmulKind::UnitPartitioned => UnitPartitioned::cubedim_resource(blueprint),
            StageMatmulKind::PlanePartitioned => PlanePartitioned::cubedim_resource(blueprint),
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
            StageMatmulKind::PlanePartitioned => PlanePartitioned::validate_blueprint(blueprint)?,
            StageMatmulKind::UnitPartitioned => UnitPartitioned::validate_blueprint(blueprint)?,
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
