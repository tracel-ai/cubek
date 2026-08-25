//! The algebra an accumulation runs under: a [`Monoid`] to fold values together, and a
//! [`Semiring`] pairing one with the product a contraction forms before folding.

use cubecl::prelude::*;

/// What a monoid asks of the values it folds: ordering and arithmetic. Every bound here is one
/// the four folds need, and the set stays well below `Numeric`, which `Vector` does not have.
pub trait Foldable:
    CubePartialOrd
    + CubeAdd
    + CubeMul
    + core::ops::Add<Self, Output = Self>
    + core::ops::Mul<Self, Output = Self>
    + Sized
{
}

impl<T> Foldable for T where
    T: CubePartialOrd
        + CubeAdd
        + CubeMul
        + core::ops::Add<Self, Output = Self>
        + core::ops::Mul<Self, Output = Self>
{
}

/// An identity and an associative fold. Everything that merges values takes one: the plane
/// instructions ([`plane`](super::plane)), the register nests
/// ([`instruction`](crate::instruction::registers)), the verb that schedules them
/// ([`Tile::reduce_axis`](crate::Tile::reduce_axis)), and the drain that combines a plane's
/// partials.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Monoid {
    /// `acc + val`, identity `0`.
    Sum,
    /// `acc * val`, identity `1`.
    Prod,
    /// `max(acc, val)`, identity the lowest value of the element.
    Max,
    /// `min(acc, val)`, identity the highest value of the element.
    Min,
}

#[cube]
impl Monoid {
    /// This monoid's identity element: folding it in leaves the other operand unchanged. What a
    /// masked read past an operand's valid extent must return instead of a shared zero, since zero
    /// is `Sum`'s identity but biases `Max` toward it (any negative data), `Min` away from it (any
    /// positive data), and annihilates `Prod`.
    pub fn identity<E: Numeric>(#[comptime] monoid: Monoid) -> E {
        match comptime!(monoid) {
            Monoid::Sum => E::from_int(0),
            Monoid::Prod => E::from_int(1),
            Monoid::Max => E::min_value(),
            Monoid::Min => E::max_value(),
        }
    }

    /// Fold `rhs` into `lhs`, over scalars and lines alike.
    fn fold_of<T: Foldable>(lhs: T, rhs: T, #[comptime] monoid: Monoid) -> T {
        match comptime!(monoid) {
            Monoid::Sum => lhs + rhs,
            Monoid::Prod => lhs * rhs,
            Monoid::Max => max(lhs, rhs),
            Monoid::Min => min(lhs, rhs),
        }
    }
}

/// `monoid.fold(a, b)`, the form call sites use.
///
/// Written out rather than generated: `#[cube]` hangs a method's expansion on `{Name}Expand`,
/// and a comptime-only value has none, so the operation is an associated function above and
/// this pair forwards to it. The plain half serves the unexpanded copy of a `#[cube]` body, the
/// `__expand` half is what the macro calls. [`identity`](Monoid::identity) takes no such pair:
/// its call is comptime through and through, which the macro folds on the host, where a generic
/// element has no value to fold.
impl Monoid {
    pub fn fold<T: Foldable>(self, lhs: T, rhs: T) -> T {
        Monoid::fold_of::<T>(lhs, rhs, self)
    }

    pub fn __expand_fold_method<T: Foldable>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
    ) -> T::ExpandType {
        Monoid::__expand_fold_of::<T>(scope, lhs, rhs, self)
    }
}

/// What a contraction contracts under: the product it forms from a pair of operands, and the
/// monoid those products accumulate into.
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
    fn step_of<T: Foldable + CubePrimitive>(
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
    pub fn step<T: Foldable + CubePrimitive>(self, lhs: T, rhs: T, acc: T) -> T {
        Semiring::step_of::<T>(lhs, rhs, acc, self)
    }

    pub fn __expand_step_method<T: Foldable + CubePrimitive>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
        acc: T::ExpandType,
    ) -> T::ExpandType {
        Semiring::__expand_step_of::<T>(scope, lhs, rhs, acc, self)
    }
}

/// A [`Monoid`] is comptime-only: a kernel never holds one in a register, it reads one to decide
/// which instruction to emit. Expanding as itself is what lets a [`CubeType`] carry one in a
/// `#[cube(comptime)]` field, the way an accumulator's scope carries the algebra it folds under.
/// Each impl below is one `CubeType` requires of an expand type; none is spare.
impl CubeType for Monoid {
    type ExpandType = Self;
}

impl IntoExpand for Monoid {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self {
        self
    }
}

impl IntoMut for Monoid {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl ExpandTypeClone for Monoid {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl CubeDebug for Monoid {}

impl AsRefExpand for Monoid {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl AsMutExpand for Monoid {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}
