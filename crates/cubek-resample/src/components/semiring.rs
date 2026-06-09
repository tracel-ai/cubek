use crate::definition::Semiring;
use cubecl::prelude::*;

/// Semiring identity element.
#[cube]
pub fn semiring_identity<C: Float>(#[comptime] s: Semiring) -> C {
    match s {
        Semiring::Linear => C::new(0.0),
        Semiring::Tropical => C::min_value(),
        Semiring::Log => C::min_value(),
    }
}

/// Combine a value with its tap weight, per the semiring.
#[cube]
pub fn semiring_combine<C: Float>(#[comptime] s: Semiring, value: C, weight: C) -> C {
    match s {
        Semiring::Linear => value * weight,
        Semiring::Tropical => value + weight,
        Semiring::Log => value + weight,
    }
}

/// Reduce: fold a new combined value into the accumulator.
#[cube]
pub fn semiring_reduce<C: Float>(#[comptime] s: Semiring, acc: C, combined: C) -> C {
    match s {
        Semiring::Linear => acc + combined,
        Semiring::Tropical => C::max(acc, combined),
        Semiring::Log => {
            let m = C::max(acc, combined);
            let diff = C::abs(acc - combined);
            m + C::log1p(C::exp(C::new(0.0) - diff))
        }
    }
}
