use crate::registry::CatalogEntry;

pub struct UnaryProblem {
    pub shape: Vec<usize>,
}

pub fn problems() -> Vec<CatalogEntry<UnaryProblem>> {
    vec![CatalogEntry::new(
        "3d_32x512x2048",
        "3D (32x512x2048)",
        UnaryProblem {
            shape: vec![32, 512, 2048],
        },
    )]
}
