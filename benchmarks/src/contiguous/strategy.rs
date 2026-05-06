use crate::registry::CatalogEntry;

pub struct ContiguousStrategy;

pub fn strategies() -> Vec<CatalogEntry<ContiguousStrategy>> {
    vec![CatalogEntry::new(
        "default",
        "Default (into_contiguous)",
        ContiguousStrategy,
    )]
}
