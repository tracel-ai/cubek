//! Values a tile holds apart from the element their type names, each named again where it is
//! used: a value of a type naming one element ([`Raw`]: a slice, registers, a tensor map), an
//! erased call ([`RawCall`]), a cmma fragment ([`RawMatrix`]) and a warpgroup accumulator
//! ([`RawAccumulator`]).

use std::marker::PhantomData;

use cubecl::cmma::{Matrix, MatrixExpand, MatrixIdent, Plane};
use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::pliron::value::Value;
use cubecl::ir::{ElemType, ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::std::tensor::{ErasedTensor, ErasedTensorExpand, ErasedTensorRaw};
use cubecl::unexpanded;
use cubecl::wgmma::{Accumulator, PendingAccumulatorRaw};

/// Which type a [`Raw`] value is, at whatever element it holds.
pub(crate) trait RawKind: 'static {
    /// The type at element `T`.
    type At<T: Numeric>: CubeType<ExpandType = NativeExpand<Self::At<T>>>;
}

/// A slice of a buffer.
pub(crate) struct Slice;
/// A slice of shared memory.
pub(crate) struct SharedSlice;
/// A unit's registers, in lines of `W`.
pub(crate) struct Registers<W: Size>(PhantomData<W>);
/// A TMA tensor map of tiled boxes.
pub(crate) struct TiledMap;

impl RawKind for Slice {
    type At<T: Numeric> = Box<[T]>;
}

impl RawKind for SharedSlice {
    type At<T: Numeric> = Shared<[T]>;
}

impl<W: Size> RawKind for Registers<W> {
    type At<T: Numeric> = Array<Vector<T, W>>;
}

impl RawKind for TiledMap {
    type At<T: Numeric> = TensorMap<T, Tiled>;
}

/// A value of the type `K` names, held as the value and the element it holds.
pub(crate) struct Raw<K: RawKind> {
    _kind: PhantomData<K>,
}

pub(crate) struct RawExpand<K: RawKind> {
    value: ExpandValue,
    held: ElemType,
    _kind: PhantomData<K>,
}

impl<K: RawKind> Raw<K> {
    /// `value`, its element kept beside it.
    pub(crate) fn of<T: Numeric>(_value: K::At<T>) -> Raw<K> {
        unexpanded!()
    }

    /// The value, named at `T`, the element it holds.
    pub(crate) fn at<T: Numeric>(&self) -> &K::At<T> {
        unexpanded!()
    }

    /// The mutable twin of [`at`](Self::at).
    pub(crate) fn at_mut<T: Numeric>(&mut self) -> &mut K::At<T> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        value: NativeExpand<K::At<T>>,
    ) -> RawExpand<K> {
        RawExpand {
            value: value.expand,
            held: T::elem_type(scope),
            _kind: PhantomData,
        }
    }
}

impl<K: RawKind> RawExpand<K> {
    pub(crate) fn __expand_at_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a NativeExpand<K::At<T>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    pub(crate) fn __expand_at_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut NativeExpand<K::At<T>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    /// The value named at `T`, which must be the element it holds.
    fn typed<T: Numeric>(&self, scope: &Scope) -> NativeExpand<K::At<T>> {
        let named = T::elem_type(scope);
        assert_eq!(
            self.held, named,
            "Raw: the value holds {:?}, not {named:?}",
            self.held
        );
        self.value.into()
    }
}

impl Raw<Slice> {
    /// Free the shared memory this slice is a window of, which the allocation traces back to its
    /// declaration.
    pub(crate) fn free(&self) {
        unexpanded!()
    }
}

impl RawExpand<Slice> {
    pub(crate) fn __expand_free_method(&self, scope: &Scope) {
        // The list a slice is a window of is the same whatever element the slice is named at.
        let slice: NativeExpand<Box<[u32]>> = self.value.into();
        let list = slice.__extract_list(scope);
        scope.register(&cubecl::ir::dialect::general::FreeOp::new(
            scope.ctx_mut(),
            list,
        ));
    }
}

/// An erased call, read from or written to, held apart from its element.
pub(crate) struct RawCall<IO> {
    _io: PhantomData<IO>,
}

pub(crate) struct RawCallExpand<IO> {
    raw: ErasedTensorRaw<IO>,
}

impl<IO> Clone for RawCall<IO> {
    fn clone(&self) -> Self {
        RawCall { _io: PhantomData }
    }
}

impl<IO> Clone for RawCallExpand<IO> {
    fn clone(&self) -> Self {
        RawCallExpand {
            raw: self.raw.clone(),
        }
    }
}

impl<IO: 'static> RawCall<IO> {
    /// `call`, its element kept beside it.
    pub(crate) fn of<T: Numeric>(_call: ErasedTensor<T, IO>) -> RawCall<IO> {
        unexpanded!()
    }

    /// The call, named at `T`, the element it holds.
    pub(crate) fn call_mut<T: Numeric>(&mut self) -> &mut ErasedTensor<T, IO> {
        unexpanded!()
    }

    /// The read-only twin of [`call_mut`](Self::call_mut).
    pub(crate) fn call<T: Numeric>(&self) -> &ErasedTensor<T, IO> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        call: ErasedTensorExpand<T, IO>,
    ) -> RawCallExpand<IO> {
        RawCallExpand {
            raw: call.into_raw(scope),
        }
    }
}

impl<IO: 'static> RawCallExpand<IO> {
    pub(crate) fn __expand_call_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut ErasedTensorExpand<T, IO> {
        scope.create_kernel_ref(ErasedTensorExpand::from_raw(scope, self.raw.clone()))
    }

    pub(crate) fn __expand_call_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a ErasedTensorExpand<T, IO> {
        scope.create_kernel_ref(ErasedTensorExpand::from_raw(scope, self.raw.clone()))
    }
}

/// A cmma fragment, held as its variable and the operand it is.
#[derive(Clone)]
pub(crate) struct RawMatrix;

#[derive(Clone)]
pub(crate) struct RawMatrixExpand {
    elem: Value,
    ident: MatrixIdent,
}

impl RawMatrix {
    /// `matrix`, its element kept in its variable's type.
    pub(crate) fn of<T: Numeric>(_matrix: Matrix<T>) -> RawMatrix {
        unexpanded!()
    }

    /// The fragment, named at `T`, the element it holds.
    pub(crate) fn matrix<T: Numeric>(&self) -> &Matrix<T> {
        unexpanded!()
    }

    /// The mutable twin of [`matrix`](Self::matrix).
    pub(crate) fn matrix_mut<T: Numeric>(&mut self) -> &mut Matrix<T> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        _scope: &Scope,
        matrix: MatrixExpand<T, Plane>,
    ) -> RawMatrixExpand {
        let (elem, ident) = matrix.into_raw();
        RawMatrixExpand { elem, ident }
    }
}

impl RawMatrixExpand {
    pub(crate) fn __expand_matrix_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a MatrixExpand<T, Plane> {
        scope.create_kernel_ref(MatrixExpand::from_raw(scope, self.elem, self.ident))
    }

    pub(crate) fn __expand_matrix_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut MatrixExpand<T, Plane> {
        scope.create_kernel_ref(MatrixExpand::from_raw(scope, self.elem, self.ident))
    }
}

/// A warpgroup accumulator handed to its MMAs, held apart from its element.
#[derive(Clone)]
pub(crate) struct RawAccumulator;

#[derive(Clone)]
pub(crate) struct RawAccumulatorExpand {
    raw: PendingAccumulatorRaw,
}

impl RawAccumulator {
    /// `acc`, its element kept beside it.
    pub(crate) fn of<T: Numeric>(_acc: Pending<Accumulator<T>>) -> RawAccumulator {
        unexpanded!()
    }

    /// The accumulator, named at `T`, the element it holds.
    pub(crate) fn pending<T: Numeric>(&self) -> &Pending<Accumulator<T>> {
        unexpanded!()
    }

    /// The mutable twin of [`pending`](Self::pending).
    pub(crate) fn pending_mut<T: Numeric>(&mut self) -> &mut Pending<Accumulator<T>> {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        acc: PendingExpand<Accumulator<T>>,
    ) -> RawAccumulatorExpand {
        RawAccumulatorExpand {
            raw: acc.into_raw(scope),
        }
    }
}

impl RawAccumulatorExpand {
    pub(crate) fn __expand_pending_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a PendingExpand<Accumulator<T>> {
        scope.create_kernel_ref(PendingExpand::from_raw(scope, self.raw))
    }

    pub(crate) fn __expand_pending_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut PendingExpand<Accumulator<T>> {
        scope.create_kernel_ref(PendingExpand::from_raw(scope, self.raw))
    }
}

impl<K: RawKind> Clone for Raw<K> {
    fn clone(&self) -> Self {
        Raw { _kind: PhantomData }
    }
}

impl<K: RawKind> Clone for RawExpand<K> {
    fn clone(&self) -> Self {
        RawExpand {
            value: self.value,
            held: self.held,
            _kind: PhantomData,
        }
    }
}

impl<K: RawKind> CubeType for Raw<K> {
    type ExpandType = RawExpand<K>;
}

impl<IO: 'static> CubeType for RawCall<IO> {
    type ExpandType = RawCallExpand<IO>;
}

impl CubeType for RawMatrix {
    type ExpandType = RawMatrixExpand;
}

impl CubeType for RawAccumulator {
    type ExpandType = RawAccumulatorExpand;
}

macro_rules! expand_value {
    ($ty: ty $(, $param: ident: $bound: tt)?) => {
        impl$(<$param: $bound>)? IntoExpand for $ty {
            type Expand = Self;
            fn into_expand(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: $bound>)? ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }
        impl$(<$param: $bound>)? IntoMut for $ty {
            fn into_mut(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: $bound>)? CubeDebug for $ty {}
        impl$(<$param: $bound>)? AsRefExpand for $ty {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl$(<$param: $bound>)? AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
    };
}

expand_value!(RawExpand<K>, K: RawKind);
expand_value!(RawCallExpand<IO>, IO: 'static);
expand_value!(RawMatrixExpand);
expand_value!(RawAccumulatorExpand);
