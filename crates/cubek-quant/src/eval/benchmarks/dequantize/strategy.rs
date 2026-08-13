use cubek_test_utils::CatalogEntry;

/// Which dequantize implementation a run measures, the two arms `dequantize::launch_ref`
/// routes between.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum DequantizePath {
    /// The legacy elementwise kernels, one scale level only.
    Legacy,
    /// The tile-engine kernel behind `dequantize_tiled`.
    Tile,
}

pub fn strategies() -> Vec<CatalogEntry<DequantizePath>> {
    vec![
        CatalogEntry::new(
            "legacy",
            "legacy elementwise kernel",
            DequantizePath::Legacy,
        ),
        CatalogEntry::new("tile", "tile engine", DequantizePath::Tile),
    ]
}
