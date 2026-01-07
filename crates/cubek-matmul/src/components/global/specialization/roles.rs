use cubecl::prelude::*;

use crate::components::global::specialization::config::LoadFlows;
use crate::components::global::{InputLoadFlow, MaxGlobalReaderPlanes};
use crate::definition::MatmulSetupError;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Represents how many planes are used for main matmul computation and for loading-only tasks.
pub struct PlaneFlowCounts {
    /// Number of planes participating in main matmul and (possibly) loading.
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
/// Contains the number of plane in each role and the rule to distinguish planes based on their plane id
pub struct PlaneFlowConfig {
    pub counts: PlaneFlowCounts,
    pub partition_rule: PlaneFlowPartitionRule,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Comptime version of [RoleRule]
pub enum PlaneFlowPartitionRule {
    MainFlowOnly,
    LoadOnlyFirst { load_only: u32 },
    LoadOnlyLast { main_flow: u32 },
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Threshold of plane id at which the roles change
///
/// Note: this struct is only necessary because Cube enums cannot hold
/// a comptime value directly
pub struct PartitionThreshold {
    #[cube(comptime)]
    threshold: u32,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Rule to distinguish a plane's role based on its plane id
pub enum PlaneFlowPartition {
    /// All planes are in the main flow, this is equivalent of having no specialization
    MainFlowOnly,
    /// Load-only planes: [0, Threshold)
    /// Main flow planes: [Threshold, total)
    LoadOnlyFirst(PartitionThreshold),
    /// Main flow planes: [0, Threshold)
    /// Load-only planes: [Threshold, total)
    LoadOnlyLast(PartitionThreshold),
}

impl PlaneFlowConfig {
    /// Make a new PlaneFlowConfig
    pub fn new(
        load_flows: LoadFlows,
        reader_tasks: Option<MaxGlobalReaderPlanes>,
        num_main_flow_planes: u32,
    ) -> Result<PlaneFlowConfig, MatmulSetupError> {
        let counts = match reader_tasks {
            Some(reader_tasks) => {
                load_flows.to_plane_flow_counts(num_main_flow_planes, reader_tasks)
            }

            None => {
                if load_flows.has_specialization() {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(
                        "Error: Load specialization config has specialization but no reader tasks were given."
                            .to_string(),
                    )));
                } else {
                    PlaneFlowCounts {
                        main_flow: num_main_flow_planes,
                        load_only: 0,
                    }
                }
            }
        };

        // TODO make possible to select LoadOnlyLast
        let rule = match counts.load_only {
            0 => PlaneFlowPartitionRule::MainFlowOnly,
            _ => PlaneFlowPartitionRule::LoadOnlyFirst {
                load_only: counts.load_only,
            },
        };

        Ok(Self {
            counts,
            partition_rule: rule,
        })
    }

    pub fn new_unspecialized(num_planes: u32) -> PlaneFlowConfig {
        PlaneFlowConfig {
            counts: PlaneFlowCounts {
                main_flow: num_planes,
                load_only: 0,
            },
            partition_rule: PlaneFlowPartitionRule::MainFlowOnly,
        }
    }

    /// Returns the number of planes participating in main flow
    pub fn main_flow_count(&self) -> u32 {
        self.counts.main_flow
    }

    /// Whether the plane role config implies specialization
    pub fn has_specialization(&self) -> bool {
        self.counts.load_only > 0
    }
}

#[cube]
impl PlaneFlowPartition {
    /// Make a cube role rule from comptime config
    pub fn new(#[comptime] comptime_rule: PlaneFlowPartitionRule) -> PlaneFlowPartition {
        match comptime!(comptime_rule) {
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
    /// ignoring load-only planes
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

    /// Whether this unit is the leader of the loading units. Will always be the lowest unit in the
    /// correct group.
    ///
    /// Only used with TMA, so has some CUDA optimizations. `plane_broadcast` and `plane_elect`
    /// ensure the compiler recognizes the values as warp uniform.
    pub fn elect_load_leader(self) -> bool {
        let plane_id = plane_broadcast(UNIT_POS_Y, 0);

        let is_elected_plane = match self {
            PlaneFlowPartition::MainFlowOnly | PlaneFlowPartition::LoadOnlyFirst(_) => {
                plane_id == 0
            }
            PlaneFlowPartition::LoadOnlyLast(main_flow) => plane_id == main_flow.threshold,
        };

        is_elected_plane && plane_elect()
    }

    /// Whether the current plane is a load-only plane
    pub fn is_load_plane(self) -> bool {
        match self {
            PlaneFlowPartition::MainFlowOnly => false,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => UNIT_POS_Y < load_only.threshold,
            PlaneFlowPartition::LoadOnlyLast(main_flow) => UNIT_POS_Y >= main_flow.threshold,
        }
    }

    /// Whether this plane is part of the compute planes
    ///
    /// Only used in specialized, so has some CUDA optimizations. `plane_broadcast` ensure the
    /// compiler recognizes the values as warp uniform.
    pub fn is_compute_plane(self) -> bool {
        let plane_id = plane_broadcast(UNIT_POS_Y, 0);

        match self {
            PlaneFlowPartition::MainFlowOnly => true,
            PlaneFlowPartition::LoadOnlyFirst(load_only) => plane_id >= load_only.threshold,
            PlaneFlowPartition::LoadOnlyLast(main_flow) => plane_id < main_flow.threshold,
        }
    }
}
