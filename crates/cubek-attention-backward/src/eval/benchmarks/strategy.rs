use cubek_test_utils::CatalogEntry;

/// What runs a row: one routine at its lead setting, or the tuned pass.
#[derive(Debug, Clone, Copy)]
pub enum AttentionBackwardStrategy {
    UnitLead,
    PlaneLead,
    CubeLead,
    Tuned,
}

pub fn strategies() -> Vec<CatalogEntry<AttentionBackwardStrategy>> {
    vec![
        CatalogEntry::new(
            "unit_lead",
            "Unit routine (lead)",
            AttentionBackwardStrategy::UnitLead,
        ),
        CatalogEntry::new(
            "plane_lead",
            "Plane fragment routine (lead)",
            AttentionBackwardStrategy::PlaneLead,
        ),
        CatalogEntry::new(
            "cube_lead",
            "Cube staged routine (lead)",
            AttentionBackwardStrategy::CubeLead,
        ),
        CatalogEntry::new("tuned", "Tuned", AttentionBackwardStrategy::Tuned),
    ]
}
