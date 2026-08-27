use super::InterpolateMode;
use cubek_tile::Boundary;

/// Filter behavior shared by the forward kernel and the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeProperties {
    pub taps: usize,
    pub renormalizes: bool,
    pub boundary: Boundary,
}

pub const fn mode_properties(mode: InterpolateMode) -> ModeProperties {
    match mode {
        InterpolateMode::Nearest(_) => ModeProperties {
            taps: 1,
            renormalizes: false,
            boundary: Boundary::Clamp,
        },
        InterpolateMode::Bilinear => ModeProperties {
            taps: 2,
            renormalizes: false,
            boundary: Boundary::Clamp,
        },
        InterpolateMode::Bicubic => ModeProperties {
            taps: 4,
            renormalizes: false,
            boundary: Boundary::Clamp,
        },
        InterpolateMode::Lanczos3 => ModeProperties {
            taps: 6,
            renormalizes: true,
            boundary: Boundary::Zero,
        },
    }
}
