use cubek_test_utils::CatalogEntry;

use crate::kernels::{ReduceKernel, ReduceKernelHeuristic, ReduceOpStrategy};

/// What separates the words of a strategy's name, which an id separates by an underscore.
const KEBAB_SEPARATOR: char = '-';

/// Every strategy: the autotune, the rule, and each kernel pinned.
pub fn strategies() -> Vec<CatalogEntry<ReduceOpStrategy>> {
    core::iter::once(ReduceOpStrategy::Autotune)
        .chain([ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile)])
        .chain(
            ReduceKernel::all()
                .into_iter()
                .map(ReduceOpStrategy::Specific),
        )
        .map(|strategy| {
            let id = strategy.name().replace(KEBAB_SEPARATOR, "_");
            CatalogEntry::new(id, strategy.name(), strategy)
        })
        .collect()
}
