use cubek_test_utils::CatalogEntry;

/// What runs a row: the cube routine at its lead setting, or the tuned walk.
#[derive(Debug, Clone, Copy)]
pub enum RecurrenceWalkStrategy {
    CubeLead,
    Tuned,
}

pub fn strategies() -> Vec<CatalogEntry<RecurrenceWalkStrategy>> {
    vec![
        CatalogEntry::new(
            "cube_lead",
            "Cube routine (lead)",
            RecurrenceWalkStrategy::CubeLead,
        ),
        CatalogEntry::new("tuned", "Tuned", RecurrenceWalkStrategy::Tuned),
    ]
}
