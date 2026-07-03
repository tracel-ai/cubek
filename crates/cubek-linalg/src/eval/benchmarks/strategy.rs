use cubek_test_utils::CatalogEntry;

use crate::launch::QRStrategy;

pub struct QrStrategy(pub QRStrategy);

pub fn strategies() -> Vec<CatalogEntry<QrStrategy>> {
    vec![
        CatalogEntry::new(
            "baht_tsqr",
            "BahtTsqr",
            QrStrategy(QRStrategy::BahtTsqr),
        ),
        CatalogEntry::new(
            "baht",
            "BAHT",
            QrStrategy(QRStrategy::BlockedAcceleratedHouseHolder),
        ),
        CatalogEntry::new(
            "cgr",
            "CGR",
            QrStrategy(QRStrategy::CommonGivensRotations),
        ),
        CatalogEntry::new(
            "mgs",
            "MGS",
            QrStrategy(QRStrategy::ModifiedGramSchmidt),
        ),
    ]
}
