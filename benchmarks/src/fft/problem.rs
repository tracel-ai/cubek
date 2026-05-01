use cubek::fft::FftMode;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_FORWARD_2K: &str = "forward_5x2x2048";
pub const PROBLEM_INVERSE_2K: &str = "inverse_5x2x2048";

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
        _ => return None,
    })
}
