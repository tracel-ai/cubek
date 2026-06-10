use crate::definition::Semiring;
use cubecl::prelude::*;

/// Semiring identity element.
#[cube]
pub fn semiring_identity<F: Float, N: Size>(#[comptime] s: &Semiring) -> Vector<F, N> {
    match s {
        Semiring::Linear => Vector::new(F::new(0.0)),
        Semiring::Tropical => Vector::min_value(),
        Semiring::Log => Vector::min_value(),
    }
}

/// Combine a value with its tap weight, per the semiring.
#[cube]
pub fn semiring_combine<F: Float, N: Size>(
    #[comptime] s: &Semiring,
    value: Vector<F, N>,
    weight: Vector<F, N>,
) -> Vector<F, N> {
    match s {
        Semiring::Linear => value * weight,
        Semiring::Tropical => value + weight,
        Semiring::Log => value + weight,
    }
}

/// Reduce: fold a new combined value into the accumulator.
#[cube]
pub fn semiring_reduce<F: Float, N: Size>(
    #[comptime] s: &Semiring,
    acc: Vector<F, N>,
    combined: Vector<F, N>,
) -> Vector<F, N> {
    match s {
        Semiring::Linear => acc + combined,
        Semiring::Tropical => acc.max(combined),
        Semiring::Log => {
            let m = acc.max(combined);
            let diff = (acc - combined).abs();
            m + (Vector::new(F::new(0.0)) - diff).exp().log1p()
        }
    }
}
