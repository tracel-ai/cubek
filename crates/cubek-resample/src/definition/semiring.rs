use cubecl::prelude::*;

/// The semiring, it determines how the values are combined.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, CubeType)]
pub enum Semiring {
    /// Linear algebra: `y = A·x`.
    Linear,
    /// Tropical algebra: `f(x, w) = x + w`.
    Tropical,
    /// Log-sum-exp algebra: `f(x, w) = log(exp(x) + exp(w))`.
    Log,
}

#[cube]
impl Semiring {
    /// Get the identity element for the semiring.
    pub fn identity<F: Float, N: Size>(#[comptime] this: &Self) -> Vector<F, N> {
        match this {
            Semiring::Linear => Vector::new(F::new(0.0)),
            Semiring::Tropical | Semiring::Log => Vector::min_value(),
        }
    }

    /// Combine a value with its weight.
    pub fn combine<F: Float, N: Size>(
        value: Vector<F, N>,
        weight: Vector<F, N>,
        #[comptime] this: &Self,
    ) -> Vector<F, N> {
        match this {
            Semiring::Linear => value * weight,
            Semiring::Tropical | Semiring::Log => value + weight,
        }
    }

    /// Accumulate the a new value in the accumulator.
    pub fn accumulate<F: Float, N: Size>(
        accumulator: Vector<F, N>,
        value: Vector<F, N>,
        #[comptime] this: &Self,
    ) -> Vector<F, N> {
        match this {
            Semiring::Linear => accumulator + value,
            Semiring::Tropical => accumulator.max(value),
            Semiring::Log => {
                let m = accumulator.max(value);
                let diff = (accumulator - value).abs();
                let zero = Vector::new(F::new(0.0));
                m + (zero - diff).exp().log1p()
            }
        }
    }
}
