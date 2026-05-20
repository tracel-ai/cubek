use cubek_test_utils::CatalogEntry;

use crate::launch::Strategy;
use crate::routines::{BlueprintStrategy, gemm::GemmStrategy, simple::SimpleArgs};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "gemm",
            "Gemm",
            Strategy::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
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
