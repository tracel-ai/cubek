//! The accumulator's cells served at the element its sum runs in.
//!
//! A cell summed in its own element stops growing once a product falls under half its spacing:
//! an `f16` sum of values near one goes no further than 2048. So a half-precision accumulator
//! sums in `f32`, and where the sum runs wider than the cells the cast is a fact of the view,
//! the way a packed operand's unpacking is ([`PackedView`](super::PackedView)): the leaf reads
//! and writes the element it contracts in and never casts.

use std::marker::PhantomData;

use cubecl::ir::VectorSize;
use cubecl::prelude::barrier::Barrier;
use cubecl::unexpanded;
use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut, ViewMutExpand, ViewOperations, ViewOperationsExpand, ViewOperationsMut,
        ViewOperationsMutExpand, layout::Coordinates,
    },
};

/// A [`ViewMut`] over cells stored at `S`, read and written as `Vector<E, V>`: widened on the
/// way out, narrowed on the way in, and touched nowhere else. The mutable twin of the packed
/// read, for the accumulating side.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct CastViewMut<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> {
    stored: ViewMut<'a, Vector<S, V>, C>,
    #[cube(comptime)]
    _served: PhantomData<E>,
}

#[cube]
impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> CastViewMut<'a, S, E, V, C> {
    pub fn new(stored: ViewMut<'a, Vector<S, V>, C>) -> Self {
        CastViewMut::<'a, S, E, V, C> {
            stored,
            _served: PhantomData,
        }
    }
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> CastViewMut<'a, S, E, V, C> {
    /// This view as a plain [`ViewMut`] of served values, which is what every accumulator takes.
    pub fn view_mut(self) -> ViewMut<'a, Vector<E, V>, C> {
        unexpanded!()
    }

    pub fn __expand_view_mut(
        scope: &Scope,
        this: CastViewMutExpand<'a, S, E, V, C>,
    ) -> ViewMutExpand<'a, Vector<E, V>, C> {
        this.__expand_view_mut_method(scope)
    }
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> CastViewMutExpand<'a, S, E, V, C> {
    fn widen(
        &self,
        scope: &Scope,
        stored: NativeExpand<Vector<S, V>>,
    ) -> NativeExpand<Vector<E, V>> {
        Vector::<E, V>::__expand_cast_from::<Vector<S, V>>(scope, stored)
    }

    fn narrow(
        &self,
        scope: &Scope,
        served: NativeExpand<Vector<E, V>>,
    ) -> NativeExpand<Vector<S, V>> {
        Vector::<S, V>::__expand_cast_from::<Vector<E, V>>(scope, served)
    }

    pub fn __expand_view_mut_method(self, scope: &Scope) -> ViewMutExpand<'a, Vector<E, V>, C> {
        ViewMutExpand::new(scope, self)
    }
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> Vectorized
    for CastViewMut<'a, S, E, V, C>
{
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> VectorizedExpand
    for CastViewMutExpand<'a, S, E, V, C>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.stored.__expand_vector_size_method(scope)
    }
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> ViewOperations<Vector<E, V>, C>
    for CastViewMut<'a, S, E, V, C>
{
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> ViewOperationsExpand<Vector<E, V>, C>
    for CastViewMutExpand<'a, S, E, V, C>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let stored = self.stored.clone().__expand_read_method(scope, pos);
        self.widen(scope, stored)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let stored = self.stored.clone().__expand_read_checked_method(scope, pos);
        self.widen(scope, stored)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        mask_value: NativeExpand<Vector<E, V>>,
    ) -> NativeExpand<Vector<E, V>> {
        // The mask is a served value, so it is selected at the served element rather than
        // narrowed into the store and widened back.
        let stored = self
            .stored
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let value = self.widen(scope, stored);
        select::expand::<Vector<E, V>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let stored = self
            .stored
            .clone()
            .__expand_read_unchecked_method(scope, pos);
        self.widen(scope, stored)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: C::ExpandType,
        _end: C::ExpandType,
    ) -> &SliceExpand<Vector<E, V>> {
        panic!("CastViewMut: cells stored at another element have no raw slice of served values")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> C::ExpandType {
        self.stored.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        self.stored.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<E, V>>,
        _pos: C::ExpandType,
    ) {
        panic!("CastViewMut: a tensor map cannot cast on the fly")
    }
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a> ViewOperationsMut<Vector<E, V>, C>
    for CastViewMut<'a, S, E, V, C>
{
}

impl<'a, S: Numeric, E: Numeric, V: Size, C: Coordinates + 'a>
    ViewOperationsMutExpand<Vector<E, V>, C> for CastViewMutExpand<'a, S, E, V, C>
{
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        value: NativeExpand<Vector<E, V>>,
    ) {
        let stored = self.narrow(scope, value);
        self.stored
            .clone()
            .__expand_write_method(scope, pos, stored);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        value: NativeExpand<Vector<E, V>>,
    ) {
        let stored = self.narrow(scope, value);
        self.stored
            .clone()
            .__expand_write_checked_method(scope, pos, stored);
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        _scope: &Scope,
        _pos: C::ExpandType,
        _end: C::ExpandType,
    ) -> &mut SliceExpand<Vector<E, V>> {
        panic!("CastViewMut: cells stored at another element have no raw slice of served values")
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<Vector<E, V>>,
        _pos: C::ExpandType,
    ) {
        panic!("CastViewMut: a tensor map cannot cast on the fly")
    }
}
