use crate::{
    components::global::MaxGlobalReaderPlanes,
    components::global::specialization::config::LoadFlows, definition::MatmulSetupError,
};

// Plane-flow vocabulary now lives in cubek-std; re-export the names callers
// in this crate (and downstream crates) have always used.
pub use cubek_std::{PlaneFlowConfig, PlaneFlowCounts, PlaneFlowPartition, PlaneFlowPartitionRule};

/// Build a [`PlaneFlowConfig`] from matmul-specific load-flow inputs.
pub fn make_plane_flow_config(
    load_flows: LoadFlows,
    reader_tasks: Option<MaxGlobalReaderPlanes>,
    num_main_flow_planes: u32,
) -> Result<PlaneFlowConfig, MatmulSetupError> {
    let counts = match reader_tasks {
        Some(reader_tasks) => load_flows.to_plane_flow_counts(num_main_flow_planes, reader_tasks),

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

    Ok(PlaneFlowConfig {
        counts,
        partition_rule: rule,
    })
}
