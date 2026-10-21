//! The tiled half of the `gemm` catalogue: the same problems as the multi-level
//! entries, so the two architectures are compared on one table.

use cubek_test_utils::CatalogEntry;

use crate::{
    kernels::{CubeStagedMatmulRoutineSettings, UnitRegistersMatmulRoutineSettings},
    strategy::Strategy,
    tiled::Strategy as Tiled,
};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
        CatalogEntry::new(
            "cpu_gemm",
            "CpuGemm (tile-DSL CPU)",
            Tiled::CpuGemm(UnitRegistersMatmulRoutineSettings::DEFAULT).into(),
        ),
        CatalogEntry::new(
            "cmma",
            "Cmma (tile-DSL)",
            Tiled::Cmma(CubeStagedMatmulRoutineSettings::DEFAULT).into(),
        ),
    ]
}
