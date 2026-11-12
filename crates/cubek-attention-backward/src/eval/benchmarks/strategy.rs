use cubek_test_utils::CatalogEntry;

/// What runs a row: one strategy at its lead setting, or the tuned pass.
#[derive(Debug, Clone, Copy)]
pub enum BenchTarget {
    UnitLead,
    PlaneLead,
    CubeLead,
    HypercubeLead,
    Tuned,
}

pub fn strategies() -> Vec<CatalogEntry<BenchTarget>> {
    vec![
        CatalogEntry::new("unit_lead", "Unit strategy (lead)", BenchTarget::UnitLead),
        CatalogEntry::new(
            "plane_lead",
            "Plane fragment strategy (lead)",
            BenchTarget::PlaneLead,
        ),
        CatalogEntry::new(
            "cube_lead",
            "Cube staged strategy (lead)",
            BenchTarget::CubeLead,
        ),
        CatalogEntry::new(
            "hypercube_lead",
            "Hypercube strategy (lead)",
            BenchTarget::HypercubeLead,
        ),
        CatalogEntry::new("tuned", "Tuned", BenchTarget::Tuned),
    ]
}
