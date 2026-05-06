use crate::registry::CatalogEntry;

pub struct FftStrategy;

pub fn strategies() -> Vec<CatalogEntry<FftStrategy>> {
    vec![CatalogEntry::new("default", "Default", FftStrategy)]
}
