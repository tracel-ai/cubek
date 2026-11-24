use cubek_test_utils::CatalogEntry;

use crate::kernels::ReduceOpStrategy;

/// What separates the words of a strategy's name, which an id separates by an underscore.
const KEBAB_SEPARATOR: char = '-';

/// Every strategy: the autotune, the rule, and each kernel pinned.
pub(crate) fn strategies() -> Vec<CatalogEntry<ReduceOpStrategy>> {
    ReduceOpStrategy::ALL
        .into_iter()
        .map(|strategy| {
            let id = strategy.name().replace(KEBAB_SEPARATOR, "_");
            CatalogEntry::new(id, strategy.name(), strategy)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_strategy_is_listed() {
        let listed: Vec<_> = strategies().into_iter().map(|entry| entry.value).collect();
        assert_eq!(listed, ReduceOpStrategy::ALL);
    }

    #[test]
    fn no_id_holds_a_dash() {
        assert!(strategies().iter().all(|entry| !entry.id.contains('-')));
    }
}
