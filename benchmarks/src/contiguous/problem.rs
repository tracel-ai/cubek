use crate::registry::CatalogEntry;

pub struct ContiguousProblem {
    pub shape: Vec<usize>,
    pub dims: Vec<(usize, usize)>,
}

pub fn problems() -> Vec<CatalogEntry<ContiguousProblem>> {
    vec![CatalogEntry::new(
        "4d_swap_1_2_2_3",
        "4D (16x16x512x512) swap (1,2)+(2,3)",
        ContiguousProblem {
            shape: vec![16, 16, 512, 512],
            dims: vec![(1, 2), (2, 3)],
        },
    )]
}
