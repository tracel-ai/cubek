//! Contains specialization config and runtime behaviours

mod config;
mod roles;
mod specializer;

pub use config::{
    InputLoadFlow, LoadFlows, LoadingSides, MatmulPlaneCounts, SpecializedLoadingSides,
};
pub use roles::{
    PlaneFlowConfig, PlaneFlowCounts, PlaneFlowPartition, PlaneFlowPartitionRule,
    make_plane_flow_config,
};
pub use specializer::{Specializer, SpecializerKind};
