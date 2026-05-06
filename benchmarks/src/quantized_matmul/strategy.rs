use cubek::matmul::{
    launch::Strategy,
    routines::{
        BlueprintStrategy, simple::SimpleArgs, vecmat_plane_parallel::GemvPlaneParallelStrategy,
    },
};

use crate::registry::CatalogEntry;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "gemv_plane_parallel",
            "Gemv Plane Parallel",
            Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(GemvPlaneParallelStrategy {
                target_num_planes: None,
            })),
        ),
        CatalogEntry::new(
            "simple_cyclic_cmma",
            "Simple Cyclic CMMA",
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            })),
        ),
    ]
}
