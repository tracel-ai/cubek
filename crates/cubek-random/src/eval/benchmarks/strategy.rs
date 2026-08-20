use cubek_test_utils::CatalogEntry;

pub struct RandomStrategy;

pub fn strategies() -> Vec<CatalogEntry<RandomStrategy>> {
    vec![CatalogEntry::new("current", "Current", RandomStrategy)]
}
