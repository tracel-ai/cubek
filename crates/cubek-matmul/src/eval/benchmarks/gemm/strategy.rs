//! Both architectures on one table: each branch owns its entries, this concatenates
//! whichever are compiled.

use cubek_test_utils::CatalogEntry;

use crate::strategy::Strategy;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    #[allow(unused_mut)]
    let mut entries = Vec::new();
    #[cfg(feature = "multi-level")]
    entries.extend(crate::multi_level::eval::gemm::strategies());
    #[cfg(feature = "tiled")]
    entries.extend(crate::tiled::eval::gemm::strategies());
    entries
}
