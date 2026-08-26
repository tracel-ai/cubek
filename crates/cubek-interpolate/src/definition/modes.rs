use super::InterpolateMode;

pub fn get_halo(mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest(_) => 1,
        InterpolateMode::Bilinear => 2,
        InterpolateMode::Bicubic => 4,
        InterpolateMode::Lanczos3 => 6,
    }
}

pub fn get_requires_bound_check(mode: InterpolateMode) -> bool {
    matches!(mode, InterpolateMode::Lanczos3)
}
