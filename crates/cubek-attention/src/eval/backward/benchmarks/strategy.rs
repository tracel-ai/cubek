use cubek_test_utils::CatalogEntry;

/// Which backward slice to benchmark. The forward-vs-backward ratio plot
/// runs both a forward (with `lse` emission) and the full backward, then
/// compares; the runner just records the backward time for now.
#[derive(Clone, Debug)]
pub enum BackwardStrategy {
    Prepass,
    Dq,
    Dkdv,
    EndToEnd,
    ForwardVsBackward,
}

pub fn strategies() -> Vec<CatalogEntry<BackwardStrategy>> {
    vec![
        CatalogEntry::new(
            "prepass",
            "Prepass (D = rowsum(dO⊙O))",
            BackwardStrategy::Prepass,
        ),
        CatalogEntry::new("dq", "dQ kernel (Q-outer)", BackwardStrategy::Dq),
        CatalogEntry::new("dkdv", "dK/dV kernel (KV-outer)", BackwardStrategy::Dkdv),
        CatalogEntry::new(
            "end_to_end",
            "End-to-end (prepass + dQ + dK/dV)",
            BackwardStrategy::EndToEnd,
        ),
        CatalogEntry::new(
            "forward_vs_backward",
            "Forward (with lse) vs. backward ratio",
            BackwardStrategy::ForwardVsBackward,
        ),
    ]
}
