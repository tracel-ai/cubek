use crate::definition::Semiring;
use cubecl::prelude::*;

/// Semiring identity element.
#[cube]
pub fn semiring_identity<F: Float>(#[comptime] s: &Semiring) -> F {
    match s {
        Semiring::Linear => F::new(0.0),
        Semiring::Tropical => F::min_value(),
        Semiring::Log => F::min_value(),
    }
}

/// Combine a value with its tap weight, per the semiring.
#[cube]
pub fn semiring_combine<F: Float>(#[comptime] s: &Semiring, value: F, weight: F) -> F {
    match s {
        Semiring::Linear => value * weight,
        Semiring::Tropical => value + weight,
        Semiring::Log => value + weight,
    }
}

/// Reduce: fold a new combined value into the accumulator.
#[cube]
pub fn semiring_reduce<F: Float>(#[comptime] s: &Semiring, acc: F, combined: F) -> F {
    match s {
        Semiring::Linear => acc + combined,
        Semiring::Tropical => F::max(acc, combined),
        Semiring::Log => {
            let m = F::max(acc, combined);
            let diff = F::abs(acc - combined);
            m + F::log1p(F::exp(F::new(0.0) - diff))
        }
    }
}
