use crate::registry::CatalogEntry;

pub struct InterpolateStrategy;

pub fn strategies() -> Vec<CatalogEntry<InterpolateStrategy>> {
    vec![CatalogEntry::new("default", "Default", InterpolateStrategy)]
}
