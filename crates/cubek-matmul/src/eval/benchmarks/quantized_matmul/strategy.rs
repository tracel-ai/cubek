use cubek_test_utils::CatalogEntry;

use crate::routines::{BlueprintStrategy, batch::simple::SimpleArgs, gemm::GemmStrategy};
use crate::strategy::Strategy;

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
