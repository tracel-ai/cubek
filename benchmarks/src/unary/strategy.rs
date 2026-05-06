use cubecl::prelude::VectorSize;

use crate::registry::CatalogEntry;

pub struct UnaryStrategy {
    pub vectorization: VectorSize,
}

pub fn strategies() -> Vec<CatalogEntry<UnaryStrategy>> {
    vec![
        CatalogEntry::new("vec1", "Vec1", UnaryStrategy { vectorization: 1 }),
        CatalogEntry::new("vec4", "Vec4", UnaryStrategy { vectorization: 4 }),
        CatalogEntry::new("vec8", "Vec8", UnaryStrategy { vectorization: 8 }),
    ]
}
