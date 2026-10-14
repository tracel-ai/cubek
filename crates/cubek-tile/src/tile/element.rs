//! The element a tile serves its values at, held as data ([`Tile::element`](crate::Tile)), and
//! named as a type where code needs one.
//!
//! A tile's ops are written once, at placeholder elements ([`Served`], [`Source`], [`Lhs`],
//! [`Rhs`], [`Out`]); an op names each placeholder as a tile's element for as long as it runs
//! ([`Naming`]), then gives it back what it named before, so an op running inside another, on a
//! tile of another element, leaves the outer op's naming as it found it.
//!
//! A placeholder is resolved wherever it is used, so a function that names one takes no element
//! generic a caller could name with it: inside, the generic would mean the function's naming, not
//! the caller's. A function that hands a tile's element down passes it as a generic instead.

use core::marker::PhantomData;

use cubecl::define_scalar;
use cubecl::ir::{ElemType, Scope};
use cubecl::prelude::*;
use cubecl::unexpanded;

// The element of the tile an op runs on.
define_scalar!(pub(crate) Served);
// The element of a tile an op reads from: a copy's source, a table, scales.
define_scalar!(pub(crate) Source);
// The elements of a contraction's two operands.
define_scalar!(pub(crate) Lhs);
define_scalar!(pub(crate) Rhs);
// The element of the tile an op writes to: a drain's destination.
define_scalar!(pub(crate) Out);

/// The placeholder `M` naming `element` until this is dropped.
pub(crate) struct Naming<'a, M: 'static> {
    scope: &'a Scope,
    before: Option<ElemType>,
    _placeholder: PhantomData<M>,
}

impl<'a, M: 'static> Naming<'a, M> {
    /// `M` names `element` from here on.
    pub(crate) fn new(scope: &'a Scope, element: ElemType) -> Self {
        let before = scope.resolve_type::<M>();
        scope.register_type::<M>(element);
        Naming {
            scope,
            before,
            _placeholder: PhantomData,
        }
    }
}

impl<M: 'static> Drop for Naming<'_, M> {
    fn drop(&mut self) {
        if let Some(before) = self.before {
            self.scope.register_type::<M>(before);
        }
    }
}

/// The placeholder `M` naming `element` for as long as the returned [`Naming`] lives: bound to a
/// name, to the end of the op that binds it.
pub(crate) fn naming<M: 'static>(_element: ElemType) -> Naming<'static, M> {
    unexpanded!()
}

pub(crate) mod naming {
    use super::*;

    pub(crate) fn expand<'a, M: 'static>(scope: &'a Scope, element: ElemType) -> Naming<'a, M> {
        Naming::new(scope, element)
    }
}

/// `T`'s element, as data.
#[cube]
// `T` is read by the expansion.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn element_of<T: Numeric>() -> comptime_type!(ElemType) {
    intrinsic!(|scope| T::elem_type(scope))
}
