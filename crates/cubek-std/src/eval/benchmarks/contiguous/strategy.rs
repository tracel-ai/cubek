use cubek_test_utils::CatalogEntry;

pub struct ContiguousStrategy;

pub fn strategies() -> Vec<CatalogEntry<ContiguousStrategy>> {
    vec![CatalogEntry::new(
        "default",
        "Default (into_contiguous)",
        ContiguousStrategy,
    )]
}
