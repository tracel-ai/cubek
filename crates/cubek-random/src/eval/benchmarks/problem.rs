use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    Uniform(f32, f32),
    Normal(f32, f32),
    Bernoulli(f32),
}

impl Distribution {
    pub fn name(&self) -> &'static str {
        match self {
            Distribution::Uniform(..) => "uniform",
            Distribution::Normal(..) => "normal",
            Distribution::Bernoulli(..) => "bernoulli",
        }
    }
}

pub struct RandomProblem {
    pub shape: Vec<usize>,
    pub distribution: Distribution,
}

pub fn problems() -> Vec<CatalogEntry<RandomProblem>> {
    let large = || vec![32, 512, 2048];
    let cache_resident = || vec![1, 2048, 2048];
    let vocab = || vec![1, 151936];
    let small = || vec![64, 64];

    vec![
        CatalogEntry::new(
            "uniform_3d_32x512x2048",
            "Uniform 3D (32x512x2048)",
            RandomProblem {
                shape: large(),
                distribution: Distribution::Uniform(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "uniform_3d_1x2048x2048",
            "Uniform 3D (1x2048x2048)",
            RandomProblem {
                shape: cache_resident(),
                distribution: Distribution::Uniform(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "normal_3d_1x2048x2048",
            "Normal 3D (1x2048x2048)",
            RandomProblem {
                shape: cache_resident(),
                distribution: Distribution::Normal(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "uniform_vocab_1x151936",
            "Uniform vocab (1x151936)",
            RandomProblem {
                shape: vocab(),
                distribution: Distribution::Uniform(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "uniform_2d_64x64",
            "Uniform 2D (64x64)",
            RandomProblem {
                shape: small(),
                distribution: Distribution::Uniform(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "normal_3d_32x512x2048",
            "Normal 3D (32x512x2048)",
            RandomProblem {
                shape: large(),
                distribution: Distribution::Normal(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "normal_vocab_1x151936",
            "Normal vocab (1x151936)",
            RandomProblem {
                shape: vocab(),
                distribution: Distribution::Normal(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "normal_2d_64x64",
            "Normal 2D (64x64)",
            RandomProblem {
                shape: small(),
                distribution: Distribution::Normal(0.0, 1.0),
            },
        ),
        CatalogEntry::new(
            "bernoulli_3d_32x512x2048",
            "Bernoulli 3D (32x512x2048)",
            RandomProblem {
                shape: large(),
                distribution: Distribution::Bernoulli(0.45),
            },
        ),
        CatalogEntry::new(
            "bernoulli_vocab_1x151936",
            "Bernoulli vocab (1x151936)",
            RandomProblem {
                shape: vocab(),
                distribution: Distribution::Bernoulli(0.45),
            },
        ),
        CatalogEntry::new(
            "bernoulli_2d_64x64",
            "Bernoulli 2D (64x64)",
            RandomProblem {
                shape: small(),
                distribution: Distribution::Bernoulli(0.45),
            },
        ),
    ]
}
