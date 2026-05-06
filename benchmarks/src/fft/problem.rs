use cubek::fft::FftMode;

use crate::registry::CatalogEntry;

pub struct FftProblem {
    pub shape: Vec<usize>,
    pub mode: FftMode,
}

pub fn problems() -> Vec<CatalogEntry<FftProblem>> {
    vec![
        CatalogEntry::new(
            "forward_5x2x2048",
            "Forward (5x2x2048)",
            FftProblem {
                shape: vec![5, 2, 2048],
                mode: FftMode::Forward,
            },
        ),
        CatalogEntry::new(
            "inverse_5x2x2048",
            "Inverse (5x2x2048)",
            FftProblem {
                shape: vec![5, 2, 2048],
                mode: FftMode::Inverse,
            },
        ),
        CatalogEntry::new(
            "forward_128x2048",
            "Forward (128x2048)",
            FftProblem {
                shape: vec![128, 2048],
                mode: FftMode::Forward,
            },
        ),
        CatalogEntry::new(
            "inverse_128x2048",
            "Inverse (128x2048)",
            FftProblem {
                shape: vec![128, 2048],
                mode: FftMode::Inverse,
            },
        ),
        CatalogEntry::new(
            "forward_1x4096",
            "Forward (1x4096)",
            FftProblem {
                shape: vec![1, 4096],
                mode: FftMode::Forward,
            },
        ),
        CatalogEntry::new(
            "inverse_1x4096",
            "Inverse (1x4096)",
            FftProblem {
                shape: vec![1, 4096],
                mode: FftMode::Inverse,
            },
        ),
        CatalogEntry::new(
            "forward_1x8192",
            "Forward (1x8192)",
            FftProblem {
                shape: vec![1, 8192],
                mode: FftMode::Forward,
            },
        ),
        CatalogEntry::new(
            "inverse_1x8192",
            "Inverse (1x8192)",
            FftProblem {
                shape: vec![1, 8192],
                mode: FftMode::Inverse,
            },
        ),
        CatalogEntry::new(
            "forward_1x16384",
            "Forward (1x16384)",
            FftProblem {
                shape: vec![1, 16384],
                mode: FftMode::Forward,
            },
        ),
        CatalogEntry::new(
            "inverse_1x16384",
            "Inverse (1x16384)",
            FftProblem {
                shape: vec![1, 16384],
                mode: FftMode::Inverse,
            },
        ),
    ]
}
