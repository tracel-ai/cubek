//! An identity and an associative operation: what every merge of values runs under.

use cubecl::prelude::*;

/// What a monoid asks of the values it combines: ordering and arithmetic. Every bound here is
/// one the four operations need, and the set stays well below `Numeric`, which `Vector` does not have.
pub trait Carrier:
    CubePartialOrd
    + CubeAdd
    + CubeMul
    + core::ops::Add<Self, Output = Self>
    + core::ops::Mul<Self, Output = Self>
    + Sized
{
}

impl<T> Carrier for T where
    T: CubePartialOrd
        + CubeAdd
        + CubeMul
        + core::ops::Add<Self, Output = Self>
        + core::ops::Mul<Self, Output = Self>
{
}

/// An identity and an associative operation, taken by everything that merges values: the plane
/// reductions, the register nests, the reduce verb, and the drain that combines a plane's partials.
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
    /// This monoid's identity element, which a masked read past an operand's valid extent must
    /// return instead of a shared zero: zero is `Sum`'s identity but biases `Max` and `Min` toward
    /// it and annihilates `Prod`.
    pub fn identity<E: Numeric>(#[comptime] monoid: Monoid) -> E {
        match comptime!(monoid) {
            Monoid::Sum => E::from_int(0),
            Monoid::Prod => E::from_int(1),
            Monoid::Max => E::min_value(),
            Monoid::Min => E::max_value(),
        }
    }

    /// `lhs ∗ rhs`, this monoid's operation: pointwise on vectors, which form the same monoid
    /// component by component.
    fn combine_of<T: Carrier>(lhs: T, rhs: T, #[comptime] monoid: Monoid) -> T {
        match comptime!(monoid) {
            Monoid::Sum => lhs + rhs,
            Monoid::Prod => lhs * rhs,
            Monoid::Max => max(lhs, rhs),
            Monoid::Min => min(lhs, rhs),
        }
    }

    /// `v₀ ∗ v₁ ∗ … ∗ v_{width-1}`: a vector's first `width` components combined into one value,
    /// starting from the identity.
    pub fn reduce<E: Numeric, N: Size>(
        v: Vector<E, N>,
        #[comptime] width: usize,
        #[comptime] monoid: Monoid,
    ) -> E {
        let mut acc = Monoid::identity::<E>(monoid);
        #[unroll]
        for j in 0..width {
            acc = monoid.combine::<E>(acc, v.extract(j));
        }
        acc
    }

    /// `seed ∗ arr₀ ∗ … ∗ arr_{len-1}`: `len` elements of `arr` combined into `seed`.
    pub fn reduce_array<E: Numeric>(
        arr: &Array<E>,
        #[comptime] len: usize,
        seed: E,
        #[comptime] monoid: Monoid,
    ) -> E {
        let mut acc = seed;
        #[unroll]
        for i in 0..len {
            acc = monoid.combine::<E>(acc, arr[i]);
        }
        acc
    }
}

/// `monoid.combine(a, b)`, the form call sites use.
///
/// Written out rather than generated: `#[cube]` hangs a method's expansion on `{Name}Expand`,
/// which a comptime-only value lacks, so the operation is an associated function above and this
/// pair forwards to it (the plain half for unexpanded `#[cube]` bodies, `__expand` for the macro).
///
/// [`identity`](Monoid::identity) needs no pair: its call is fully comptime and folds on the host.
impl Monoid {
    pub fn combine<T: Carrier>(self, lhs: T, rhs: T) -> T {
        Monoid::combine_of::<T>(lhs, rhs, self)
    }

    pub fn __expand_combine_method<T: Carrier>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
    ) -> T::ExpandType {
        Monoid::__expand_combine_of::<T>(scope, lhs, rhs, self)
    }
}

/// An algebra is comptime-only: a kernel reads one to decide which instruction to emit, never
/// holds one in a register. Expanding as itself lets a [`CubeType`] carry one in a
/// `#[cube(comptime)]` field. Each impl below is one `CubeType` requires of an expand type, the
/// same seven cubecl writes by hand for `MatrixLayout`.
macro_rules! comptime_only {
    ($ty:ty) => {
        impl CubeType for $ty {
            type ExpandType = Self;
        }

        impl IntoExpand for $ty {
            type Expand = Self;

            fn into_expand(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl IntoMut for $ty {
            fn into_mut(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                *self
            }
        }

        impl CubeDebug for $ty {}

        impl AsRefExpand for $ty {
            fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
                self
            }
        }

        impl AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
                self
            }
        }
    };
}

pub(crate) use comptime_only;

comptime_only!(Monoid);
