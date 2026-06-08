use crate::definition::Semiring;
use cubecl::prelude::*;
use std::marker::PhantomData;

#[cube]
pub trait GlobalOperation<C: Numeric>: CubeType {
    type Config: Clone + Send + Sync;

    fn identity(#[comptime] config: Self::Config) -> C;

    fn combine(value: C, weight: C) -> C;

    fn reduce(accumulator: C, combined: C, #[comptime] config: Self::Config) -> C;

    fn finalize(accumulator: C) -> C;
}

#[derive(CubeType)]
pub struct ScalarOperation<C: Numeric> {
    #[cube(comptime)]
    _marker: PhantomData<C>,
}

#[cube]
impl<C: Numeric> GlobalOperation<C> for ScalarOperation<C> {
    type Config = Semiring;

    fn identity(#[comptime] config: Self::Config) -> C {
        match config {
            Semiring::Sum => C::from_int(0),
            Semiring::Prod => C::from_int(1),
            Semiring::Max => C::min_value(),
            Semiring::Min => C::max_value(),
            Semiring::Any => C::from_int(0),
            Semiring::All => C::from_int(1),
        }
    }

    fn combine(value: C, weight: C) -> C {
        value * weight
    }

    fn reduce(accumulator: C, combined: C, #[comptime] config: Self::Config) -> C {
        match config {
            Semiring::Sum => accumulator + combined,
            Semiring::Prod => accumulator * combined,
            Semiring::Max => max(accumulator, combined),
            Semiring::Min => min(accumulator, combined),
            Semiring::Any => accumulator + combined,
            Semiring::All => accumulator + combined,
        }
    }

    fn finalize(accumulator: C) -> C {
        accumulator
    }
}
