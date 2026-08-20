use cubek_test_utils::CatalogEntry;

use crate::{PrngBlueprint, PrngStrategy};

pub struct RandomStrategy {
    pub(crate) prng: PrngStrategy,
}

pub fn strategies() -> Vec<CatalogEntry<RandomStrategy>> {
    vec![
        CatalogEntry::new(
            "auto",
            "Auto",
            RandomStrategy {
                prng: PrngStrategy::Inferred,
            },
        ),
        CatalogEntry::new(
            "interleaved",
            "Interleaved",
            RandomStrategy {
                prng: PrngStrategy::Forced(PrngBlueprint::Interleaved),
            },
        ),
        CatalogEntry::new(
            "blocked",
            "Blocked",
            RandomStrategy {
                prng: PrngStrategy::Forced(PrngBlueprint::Blocked),
            },
        ),
    ]
}
