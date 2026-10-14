//! What a [`Memory`](super::Memory) is backed by, apart from the element it holds: the raw buffer
//! ([`RawBuffer`]) or the erased call ([`RawCall`]), each typed again where it is used.

use std::marker::PhantomData;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ElemType, ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::std::tensor::{ErasedTensor, ErasedTensorExpand, ErasedTensorRaw};
use cubecl::unexpanded;

/// A buffer of values, held as the slice it was handed as and the element it holds.
#[derive(Clone)]
pub(crate) struct RawBuffer;

#[derive(Clone)]
pub(crate) struct RawBufferExpand {
    slice: ExpandValue,
    held: ElemType,
}

impl RawBuffer {
    /// `buffer`, its element kept beside it.
    #[allow(clippy::boxed_local)]
    pub(crate) fn of<T: Numeric>(_buffer: Box<[T]>) -> RawBuffer {
        unexpanded!()
    }

    /// The buffer, typed at `T`, the element it holds.
    // `Box<[T]>` is cubecl's owned-slice handle, not a Rust box.
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer<T: Numeric>(&self) -> &Box<[T]> {
        unexpanded!()
    }

    /// The mutable twin of [`buffer`](Self::buffer).
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer_mut<T: Numeric>(&mut self) -> &mut Box<[T]> {
        unexpanded!()
    }

    /// Free the shared memory this buffer is a window of, which the allocation traces back to its
    /// declaration.
    pub(crate) fn free(&self) {
        unexpanded!()
    }

    pub(crate) fn __expand_of<T: Numeric>(
        scope: &Scope,
        buffer: NativeExpand<Box<[T]>>,
    ) -> RawBufferExpand {
        RawBufferExpand {
            slice: buffer.expand,
            held: T::elem_type(scope),
        }
    }
}

impl RawBufferExpand {
    pub(crate) fn __expand_buffer_method<'a, T: Numeric>(
        &self,
        scope: &Scope,
    ) -> &'a NativeExpand<Box<[T]>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    pub(crate) fn __expand_buffer_mut_method<'a, T: Numeric>(
        &mut self,
        scope: &Scope,
    ) -> &'a mut NativeExpand<Box<[T]>> {
        scope.create_kernel_ref(self.typed::<T>(scope))
    }

    pub(crate) fn __expand_free_method(&self, scope: &Scope) {
        // The list a slice is a window of is the same whatever element the slice is named at.
        let slice: NativeExpand<Box<[u32]>> = self.slice.into();
        let list = slice.__extract_list(scope);
        scope.register(&cubecl::ir::dialect::general::FreeOp::new(
            scope.ctx_mut(),
            list,
        ));
    }

    /// The slice typed at `T`, which must be the element it holds.
    fn typed<T: Numeric>(&self, scope: &Scope) -> NativeExpand<Box<[T]>> {
        let named = T::elem_type(scope);
        assert_eq!(
            self.held, named,
            "RawBuffer: the buffer holds {:?}, not {named:?}",
            self.held
        );
        self.slice.into()
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

impl CubeType for RawBuffer {
    type ExpandType = RawBufferExpand;
}

impl<IO: 'static> CubeType for RawCall<IO> {
    type ExpandType = RawCallExpand<IO>;
}

macro_rules! expand_value {
    ($ty: ty $(, $param: ident)?) => {
        impl$(<$param: 'static>)? IntoExpand for $ty {
            type Expand = Self;
            fn into_expand(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: 'static>)? ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }
        impl$(<$param: 'static>)? IntoMut for $ty {
            fn into_mut(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$param: 'static>)? CubeDebug for $ty {}
        impl$(<$param: 'static>)? AsRefExpand for $ty {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl$(<$param: 'static>)? AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
    };
}

expand_value!(RawBufferExpand);
expand_value!(RawCallExpand<IO>, IO);
