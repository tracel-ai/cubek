//! What a normalized separable factor run divides by: which taps it sums, and how the
//! division answers a denominator too small to divide by.

use cubecl::prelude::*;

use crate::Space;

/// What a normalized factor run sums and divides by, and the space whose complete runs it
/// describes: a level below it that split a contracted axis would normalize each chunk on its
/// own, which the gather leaf refuses by comparing against `over`.
#[derive(Clone, PartialEq, Debug)]
pub struct Normalization {
    pub taps: TapSupport,
    pub guard: DivGuard,
    pub over: Space,
}

impl Normalization {
    pub fn new(taps: TapSupport, guard: DivGuard, over: Space) -> Self {
        Normalization { taps, guard, over }
    }
}

/// How division handles a denominator whose magnitude is too small to divide by. Both fields are
/// comptime kernel constants. The fallback is the reciprocal multiplier, so the default maps a
/// guarded result to zero.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DivGuard {
    pub epsilon: f32,
    pub fallback: f32,
}

impl DivGuard {
    /// A guard that falls back to `fallback` where the denominator's magnitude is at most
    /// `epsilon`. Both are kernel constants, so a bad one is caught where it is written.
    pub fn new(epsilon: f32, fallback: f32) -> Self {
        assert!(
            epsilon.is_finite() && epsilon >= 0.0,
            "DivGuard: epsilon must be finite and non-negative"
        );
        assert!(fallback.is_finite(), "DivGuard: fallback must be finite");
        DivGuard { epsilon, fallback }
    }
}

/// Which taps contribute to the sum of a normalized separable filter factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TapSupport {
    /// Sum only taps whose projected input sample is in bounds, removing edge darkening. The
    /// contraction must read the original source window in place; a shared-memory stage no longer
    /// records which staged zeros came from outside that window and is rejected when launching.
    #[default]
    InBounds,
    /// Sum the filter's whole support, preserving the fade of zero padding at an edge.
    Whole,
}

/// A guarded reciprocal that preserves the sign of a valid denominator. Substitution keeps the
/// discarded division finite even though `select` evaluates both arms; NaN fails the comparison
/// and takes the fallback.
///
/// The contraction remains generic over numeric weights for unnormalized recipes. Only the
/// float-only public normalization surface can set the flag that reaches this helper.
#[cube]
pub(crate) fn guarded_recip<E: Numeric>(d: E, #[comptime] guard: DivGuard) -> E {
    let epsilon = E::cast_from(comptime!(guard.epsilon));
    let fallback = E::cast_from(comptime!(guard.fallback));
    let valid = d.abs() > epsilon;
    let safe = select(valid, d, E::from_int(1));
    select(valid, E::from_int(1) / safe, fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_guard_accepts_finite_non_negative_thresholds() {
        DivGuard::new(0.0, 0.0);
        DivGuard::new(1.0e-7, -1.0);
    }

    #[test]
    fn division_guard_rejects_invalid_thresholds() {
        for epsilon in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| DivGuard::new(epsilon, 0.0)).is_err(),
                "epsilon {epsilon:?} should be rejected"
            );
        }
        for fallback in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| DivGuard::new(1.0e-7, fallback)).is_err(),
                "fallback {fallback:?} should be rejected"
            );
        }
    }
}
