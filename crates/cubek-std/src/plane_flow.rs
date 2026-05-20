//! Plane-flow vocabulary used by stage / global readers / partition tiles.
//! All types live here:
//!   - comptime data: [`PlaneFlowCounts`], [`PlaneFlowPartitionRule`],
//!     [`PlaneFlowConfig`], [`InputLoadFlow`]
//!   - runtime cube types: [`PartitionThreshold`], [`PlaneFlowPartition`]
//!   - composition helper: [`partition_coordinates`]
//!
//! cubek-matmul (and other consumers) re-export these to keep their own paths
//! stable; the canonical home is here so tile-level primitives can compose
//! plane-flow logic without reaching into cubek-matmul.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::Partitioner;

// ============================================================================
// Comptime data: plane counts, partition rule, full config, input-load flow.
// ============================================================================

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Represents how many planes are used for main computation and for loading-only tasks.
pub struct PlaneFlowCounts {
    /// Number of planes participating in main flow and (possibly) loading.
    pub main_flow: u32,
    /// Number of planes dedicated solely to loading.
    pub load_only: u32,
}

impl PlaneFlowCounts {
    /// Return the total number of planes
    pub fn total_count(&self) -> u32 {
        self.main_flow + self.load_only
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// How planes are partitioned by id between the main flow and load-only roles.
pub enum PlaneFlowPartitionRule {
    MainFlowOnly,
    LoadOnlyFirst { load_only: u32 },
    LoadOnlyLast { main_flow: u32 },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Plane-flow configuration carried by `CubeDimResource::Specialized`. Holds the
/// counts for main-flow vs load-only planes and the partition rule used at
/// runtime.
pub struct PlaneFlowConfig {
    pub counts: PlaneFlowCounts,
    pub partition_rule: PlaneFlowPartitionRule,
}

impl PlaneFlowConfig {
    /// All planes participate in the main flow; no load-only planes.
    pub fn new_unspecialized(num_planes: u32) -> Self {
        Self {
            counts: PlaneFlowCounts {
                main_flow: num_planes,
                load_only: 0,
            },
            partition_rule: PlaneFlowPartitionRule::MainFlowOnly,
        }
    }

    /// Number of planes participating in main flow.
    pub fn main_flow_count(&self) -> u32 {
        self.counts.main_flow
    }

    /// Whether the configuration uses dedicated load-only planes.
    pub fn has_specialization(&self) -> bool {
        self.counts.load_only > 0
    }
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Determines which types of planes are responsible for loading a tensor.
pub enum InputLoadFlow {
    /// Loaded exclusively by planes that participate in the main computation flow.
    #[default]
    MainOnly,
    /// Loaded exclusively by planes dedicated to loading (load-only planes).
    LoadOnly,
}

impl InputLoadFlow {
    /// Whether there is specialization for the tensor.
    pub fn has_specialization(&self) -> bool {
        matches!(self, InputLoadFlow::LoadOnly)
    }
}

// ============================================================================
// Runtime cube types: PartitionThreshold, PlaneFlowPartition.
// ============================================================================

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Threshold of plane id at which the roles change.
///
/// Only exists because Cube enums cannot hold a comptime value directly.
pub struct PartitionThreshold {
    #[cube(comptime)]
    threshold: u32,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Runtime view of [`PlaneFlowPartitionRule`]: distinguishes a plane's role
/// based on its plane id.
pub enum PlaneFlowPartition {
    /// All planes are in the main flow (no specialization).
    MainFlowOnly,
    /// Load-only planes: `[0, Threshold)`; main-flow planes: `[Threshold, total)`.
    LoadOnlyFirst(PartitionThreshold),
    /// Main-flow planes: `[0, Threshold)`; load-only planes: `[Threshold, total)`.
    LoadOnlyLast(PartitionThreshold),
}

#[cube]
impl PlaneFlowPartition {
    /// Construct from comptime rule.
    pub fn new(#[comptime] comptime_rule: PlaneFlowPartitionRule) -> PlaneFlowPartition {
        match comptime_rule {
            PlaneFlowPartitionRule::MainFlowOnly => PlaneFlowPartition::new_MainFlowOnly(),
            PlaneFlowPartitionRule::LoadOnlyFirst { load_only } => {
                PlaneFlowPartition::new_LoadOnlyFirst(PartitionThreshold {
                    threshold: load_only,
                })
            }
            PlaneFlowPartitionRule::LoadOnlyLast { main_flow } => {
                PlaneFlowPartition::new_LoadOnlyLast(PartitionThreshold {
                    threshold: main_flow,
                })
            }
        }
    }

    /// The index of the current plane among planes that perform compute,
    /// ignoring load-only planes.
    pub fn compute_index(self) -> u32 {
        match self {
            PlaneFlowPartition::MainFlowOnly => UNIT_POS_Y,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => UNIT_POS_Y - load_only.threshold,
            PlaneFlowPartition::LoadOnlyLast(_) => UNIT_POS_Y,
        }
    }

    /// The index of the current plane among planes that perform loading,
    /// ignoring any plane that does not participate for this `ident`.
    pub fn load_index(self, #[comptime] specialization_tensor_config: InputLoadFlow) -> u32 {
        match self {
            PlaneFlowPartition::MainFlowOnly => UNIT_POS_Y,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => match specialization_tensor_config {
                InputLoadFlow::MainOnly => UNIT_POS_Y - load_only.threshold,
                InputLoadFlow::LoadOnly => UNIT_POS_Y,
            },
            PlaneFlowPartition::LoadOnlyLast(main_flow) => match specialization_tensor_config {
                InputLoadFlow::LoadOnly => UNIT_POS_Y - main_flow.threshold,
                InputLoadFlow::MainOnly => UNIT_POS_Y,
            },
        }
    }

    /// Whether this unit is the leader of the loading units. Always the lowest
    /// unit in the correct group. Used by TMA; `plane_broadcast` / `plane_elect`
    /// keep the value warp-uniform.
    pub fn elect_load_leader(&self) -> bool {
        let plane_id = plane_broadcast(UNIT_POS_Y, 0u32);

        let is_elected_plane = match self {
            PlaneFlowPartition::MainFlowOnly | PlaneFlowPartition::LoadOnlyFirst(_) => {
                plane_id == 0
            }
            PlaneFlowPartition::LoadOnlyLast(main_flow) => plane_id == main_flow.threshold,
        };

        is_elected_plane && plane_elect()
    }

    /// Whether the current plane is a load-only plane.
    pub fn is_load_plane(self) -> bool {
        match self {
            PlaneFlowPartition::MainFlowOnly => false,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => UNIT_POS_Y < load_only.threshold,
            PlaneFlowPartition::LoadOnlyLast(main_flow) => UNIT_POS_Y >= main_flow.threshold,
        }
    }

    /// Whether this plane is part of the compute planes. Used in specialized
    /// kernels; `plane_broadcast` keeps the value warp-uniform.
    pub fn is_compute_plane(self) -> bool {
        let plane_id = plane_broadcast(UNIT_POS_Y, 0u32);

        match self {
            PlaneFlowPartition::MainFlowOnly => true,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => plane_id >= load_only.threshold,
            PlaneFlowPartition::LoadOnlyLast(main_flow) => plane_id < main_flow.threshold,
        }
    }
}

// ============================================================================
// Composition helper: combines PlaneFlowPartition::compute_index with a
// Partitioner's `coordinates` to return the current primitive's (row, col)
// in the partition grid.
// ============================================================================

#[cube]
/// Returns the `(row, col)` of the current compute primitive within the stage,
/// deriving `compute_index` from `role_rule_config` via [`PlaneFlowPartition`]
/// and delegating the per-scope math to
/// [`Partitioner::coordinates`](crate::tile::Partitioner::coordinates).
pub fn partition_coordinates<P: Partitioner>(
    #[comptime] role_rule_config: PlaneFlowPartitionRule,
    #[comptime] plane_dim: u32,
    #[comptime] num_partitions_col: u32,
) -> Coords2d {
    let compute_index = PlaneFlowPartition::new(role_rule_config).compute_index();
    P::coordinates(compute_index, plane_dim, num_partitions_col)
}
