use cubek::fft::FftMode;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_FORWARD_2K: &str = "forward_5x2x2048";
pub const PROBLEM_INVERSE_2K: &str = "inverse_5x2x2048";
pub const PROBLEM_FORWARD_MANY_2K: &str = "forward_128x2048";
pub const PROBLEM_INVERSE_MANY_2K: &str = "inverse_128x2048";
pub const PROBLEM_FORWARD_4K: &str = "forward_1x4096";
pub const PROBLEM_INVERSE_4K: &str = "inverse_1x4096";
pub const PROBLEM_FORWARD_8K: &str = "forward_1x8192";
pub const PROBLEM_INVERSE_8K: &str = "inverse_1x8192";
pub const PROBLEM_FORWARD_16K: &str = "forward_1x16384";
pub const PROBLEM_INVERSE_16K: &str = "inverse_1x16384";

pub struct FftProblem {
    pub shape: Vec<usize>,
    pub mode: FftMode,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_FORWARD_2K.to_string(),
            label: "Forward (5x2x2048)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_INVERSE_2K.to_string(),
            label: "Inverse (5x2x2048)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_FORWARD_MANY_2K.to_string(),
            label: "Forward (128x2048)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_INVERSE_MANY_2K.to_string(),
            label: "Inverse (128x2048)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_FORWARD_4K.to_string(),
            label: "Forward (1x4096)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_INVERSE_4K.to_string(),
            label: "Inverse (1x4096)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_FORWARD_8K.to_string(),
            label: "Forward (1x8192)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_INVERSE_8K.to_string(),
            label: "Inverse (1x8192)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_FORWARD_16K.to_string(),
            label: "Forward (1x16384)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_INVERSE_16K.to_string(),
            label: "Inverse (1x16384)".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<FftProblem> {
    Some(match id {
        PROBLEM_FORWARD_2K => FftProblem {
            shape: vec![5, 2, 2048],
            mode: FftMode::Forward,
        },
        PROBLEM_INVERSE_2K => FftProblem {
            shape: vec![5, 2, 2048],
            mode: FftMode::Inverse,
        },
        PROBLEM_FORWARD_MANY_2K => FftProblem {
            shape: vec![128, 2048],
            mode: FftMode::Forward,
        },
        PROBLEM_INVERSE_MANY_2K => FftProblem {
            shape: vec![128, 2048],
            mode: FftMode::Inverse,
        },
        PROBLEM_FORWARD_4K => FftProblem {
            shape: vec![1, 4096],
            mode: FftMode::Forward,
        },
        PROBLEM_INVERSE_4K => FftProblem {
            shape: vec![1, 4096],
            mode: FftMode::Inverse,
        },
        PROBLEM_FORWARD_8K => FftProblem {
            shape: vec![1, 8192],
            mode: FftMode::Forward,
        },
        PROBLEM_INVERSE_8K => FftProblem {
            shape: vec![1, 8192],
            mode: FftMode::Inverse,
        },
        PROBLEM_FORWARD_16K => FftProblem {
            shape: vec![1, 16384],
            mode: FftMode::Forward,
        },
        PROBLEM_INVERSE_16K => FftProblem {
            shape: vec![1, 16384],
            mode: FftMode::Inverse,
        },
        _ => return None,
    })
}
