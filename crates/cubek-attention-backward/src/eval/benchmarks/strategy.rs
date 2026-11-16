use cubek_test_utils::CatalogEntry;

/// What runs a row: one strategy at its lead setting, or the tuned pass.
#[derive(Debug, Clone, Copy)]
pub enum AttentionBackwardStrategy {
    UnitLead,
    PlaneLead,
    CubeLead,
    HypercubeLead,
    Tuned,
}

pub fn strategies() -> Vec<CatalogEntry<AttentionBackwardStrategy>> {
    vec![
        CatalogEntry::new(
            "unit_lead",
            "Unit strategy (lead)",
            AttentionBackwardStrategy::UnitLead,
        ),
        CatalogEntry::new(
            "plane_lead",
            "Plane fragment strategy (lead)",
            AttentionBackwardStrategy::PlaneLead,
        ),
        CatalogEntry::new(
            "cube_lead",
            "Cube staged strategy (lead)",
            AttentionBackwardStrategy::CubeLead,
        ),
        CatalogEntry::new(
            "hypercube_lead",
            "Hypercube strategy (lead)",
            AttentionBackwardStrategy::HypercubeLead,
        ),
        CatalogEntry::new("tuned", "Tuned", AttentionBackwardStrategy::Tuned),
    ]
}
