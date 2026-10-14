//! The element a tile serves its values at, held as data ([`Tile::element`](crate::Tile)), and
//! named as a type where code needs one.
//!
//! A tile's ops are written once, at a placeholder element ([`Served`], [`Other`]); an op names
//! the placeholder as the tile's element for as long as it runs ([`Naming`]), then gives the
//! placeholder back what it named before, so an op running inside another, on a tile of another
//! element, leaves the outer op's naming as it found it.

use core::marker::PhantomData;

use cubecl::define_scalar;
use cubecl::ir::{ElemType, Scope};
use cubecl::prelude::*;
use cubecl::unexpanded;

// The element of the tile an op runs on.
define_scalar!(pub(crate) Served);
// The element of the other tile an op reads: a copy's source, a contraction's operand.
define_scalar!(pub(crate) Other);

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
pub(crate) fn element_of<T: Numeric>() -> comptime_type!(ElemType) {
    intrinsic!(|scope| T::elem_type(scope))
}
