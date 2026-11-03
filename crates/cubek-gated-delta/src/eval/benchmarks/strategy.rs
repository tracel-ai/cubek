use cubek_test_utils::CatalogEntry;

/// What runs a row: the cube routine at its lead setting, or the tuned walk.
#[derive(Debug, Clone, Copy)]
pub enum GatedDeltaWalkStrategy {
    CubeLead,
    Tuned,
}

pub fn strategies() -> Vec<CatalogEntry<GatedDeltaWalkStrategy>> {
    vec![
        CatalogEntry::new(
            "cube_lead",
            "Cube routine (lead)",
            GatedDeltaWalkStrategy::CubeLead,
        ),
        CatalogEntry::new("tuned", "Tuned", GatedDeltaWalkStrategy::Tuned),
    ]
}
