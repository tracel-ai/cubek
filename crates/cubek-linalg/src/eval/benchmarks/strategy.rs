use cubek_test_utils::CatalogEntry;

/// Marker type: `cubek-linalg` only implements the `baht_tsqr` QR strategy.
pub struct QrStrategy;

pub fn strategies() -> Vec<CatalogEntry<QrStrategy>> {
    vec![CatalogEntry::new("baht_tsqr", "BahtTsqr", QrStrategy)]
}
