use cubek_test_utils::CatalogEntry;

use crate::{
    multi_level::{
        Strategy as MultiLevel,
        routines::{batch::simple::SimpleArgs, gemm::GemmStrategy},
    },
    routine::BlueprintStrategy,
    strategy::Strategy,
};

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
