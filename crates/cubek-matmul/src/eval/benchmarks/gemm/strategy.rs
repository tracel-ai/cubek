//! Both families on one table: each branch owns its entries, this concatenates
//! whichever are compiled.

use cubek_test_utils::CatalogEntry;

use crate::strategy::Strategy;

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    let mut entries = Vec::new();
    entries.extend(crate::multi_level::eval::gemm::strategies());
    entries.extend(crate::tiled::eval::gemm::strategies());
    entries
}
