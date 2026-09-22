//! What a contraction contracts under: the product it forms from a pair of operands, and the
//! monoid those products accumulate into.

use cubecl::prelude::*;

use super::monoid::{Carrier, Monoid, comptime_only};

/// A product monoid and the accumulation monoid its products fold into.
///
/// The fields are private and the pairs that are real semirings are named as constants, so a
/// combination that is not one is unsayable rather than rejected somewhere downstream.
///
/// [`add`](Semiring::add) is the half every drain reads: partials merge the same way whether they
/// came from a contraction or a reduction, which is why an accumulator's scope keeps only that.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Semiring {
    add: Monoid,
    mul: Monoid,
}

impl Semiring {
    /// `(+, *)`: the ordinary matmul.
    pub const SUM_PROD: Self = Self {
        add: Monoid::Sum,
        mul: Monoid::Prod,
    };
    /// `(min, +)`: shortest paths, DTW.
    pub const MIN_SUM: Self = Self {
        add: Monoid::Min,
        mul: Monoid::Sum,
    };
    /// `(max, +)`: Viterbi in log space.
    pub const MAX_SUM: Self = Self {
        add: Monoid::Max,
        mul: Monoid::Sum,
    };

    /// The monoid products accumulate into: what seeds an accumulator, what commits it, and what
    /// every drain folds partials under.
    pub const fn add(self) -> Monoid {
        self.add
    }

    /// The product a pair of operands forms before it is folded in.
    pub const fn mul(self) -> Monoid {
        self.mul
    }
}

#[cube]
impl Semiring {
    /// One accumulation step: `acc + (lhs * rhs)` under this semiring's two monoids. Takes its
    /// operands in `fma`'s order, which is the instruction the ordinary semiring is.
    ///
    /// One function rather than [`Monoid::fold`] twice because [`SUM_PROD`](Semiring::SUM_PROD)
    /// must stay a single `fma`: a separate multiply and dependent add doubles the FP instruction
    /// count and serializes the accumulate, since the CPU backend contracts neither.
    fn step_of<T: Carrier + CubePrimitive>(
        lhs: T,
        rhs: T,
        acc: T,
        #[comptime] semiring: Semiring,
    ) -> T {
        if comptime!(semiring == Semiring::SUM_PROD) {
            fma(lhs, rhs, acc)
        } else {
            let product = semiring.mul().fold::<T>(lhs, rhs);
            semiring.add().fold::<T>(product, acc)
        }
    }
}

/// `semiring.step(a, b, acc)`, the pair [`Monoid::fold`] documents.
impl Semiring {
    pub fn step<T: Carrier + CubePrimitive>(self, lhs: T, rhs: T, acc: T) -> T {
        Semiring::step_of::<T>(lhs, rhs, acc, self)
    }

    pub fn __expand_step_method<T: Carrier + CubePrimitive>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
        acc: T::ExpandType,
    ) -> T::ExpandType {
        Semiring::__expand_step_of::<T>(scope, lhs, rhs, acc, self)
    }
}

comptime_only!(Semiring);
