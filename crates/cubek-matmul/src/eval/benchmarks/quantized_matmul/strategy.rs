use cubek_test_utils::CatalogEntry;

use crate::multi_level::Strategy as MultiLevel;
use crate::multi_level::routines::batch::simple::SimpleArgs;
use crate::multi_level::routines::gemm::GemmStrategy;
use crate::routine::BlueprintStrategy;
use crate::strategy::Strategy;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "gemm",
            "Gemm",
            MultiLevel::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
                target_num_planes: None,
            }))
            .into(),
        ),
        CatalogEntry::new(
            "simple_cyclic_cmma",
            "Simple Cyclic CMMA",
            MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: false,
                ..Default::default()
            }))
            .into(),
        ),
    ]
}
