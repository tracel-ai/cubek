use crate::definition::Semiring;
use cubecl::prelude::*;

/// Semiring identity element.
#[cube]
pub fn semiring_identity<F: Float, N: Size>(#[comptime] semiring: &Semiring) -> Vector<F, N> {
    match semiring {
        Semiring::Linear => Vector::new(F::new(0.0)),
        Semiring::Tropical | Semiring::Log => Vector::min_value(),
    }
}

/// Combine a value with its tap weight, per the semiring.
#[cube]
pub fn semiring_combine<F: Float, N: Size>(
    value: Vector<F, N>,
    weight: Vector<F, N>,
    #[comptime] semiring: &Semiring,
) -> Vector<F, N> {
    match semiring {
        Semiring::Linear => value * weight,
        Semiring::Tropical | Semiring::Log => value + weight,
    }
}

/// Reduce a new combined value into the accumulator.
#[cube]
pub fn semiring_reduce<F: Float, N: Size>(
    acc: Vector<F, N>,
    combined: Vector<F, N>,
    #[comptime] semiring: &Semiring,
) -> Vector<F, N> {
    match semiring {
        Semiring::Linear => acc + combined,
        Semiring::Tropical => acc.max(combined),
        Semiring::Log => {
            let m = acc.max(combined);
            let diff = (acc - combined).abs();
            m + (Vector::new(F::new(0.0)) - diff).exp().log1p()
        }
    }
}
