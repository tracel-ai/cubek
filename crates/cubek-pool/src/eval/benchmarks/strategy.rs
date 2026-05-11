use cubek_test_utils::CatalogEntry;

pub struct PoolStrategy;

pub fn strategies() -> Vec<CatalogEntry<PoolStrategy>> {
    vec![CatalogEntry::new("default", "Default", PoolStrategy)]
}
