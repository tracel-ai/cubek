use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, simple::SimpleArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
    },
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_GEMV_PLANE_PARALLEL: &str = "gemv_plane_parallel";
pub const STRATEGY_SIMPLE_CYCLIC_CMMA: &str = "simple_cyclic_cmma";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_GEMV_PLANE_PARALLEL.to_string(),
            label: "Gemv Plane Parallel".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_SIMPLE_CYCLIC_CMMA.to_string(),
            label: "Simple Cyclic CMMA".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<Strategy> {
    Some(match id {
        STRATEGY_GEMV_PLANE_PARALLEL => {
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
                target_num_planes: None,
            }))
        }
        STRATEGY_SIMPLE_CYCLIC_CMMA => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
        }
        _ => return None,
    })
}
