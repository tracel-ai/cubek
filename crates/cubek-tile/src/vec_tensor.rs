//! [`VecTensor`]: a launch tensor whose *binding element type* is `Vector<E, v>`, with `v` a
//! plain launch-time value ([`VecTensorArg::new`]) — no `Size` type parameter, no kernel
//! generic.
//!
//! Why the binding type matters: re-grouping a *scalar* binding into lines in-kernel
//! (`with_vector_size`) needs the `memory_reinterpret` capability, which only CUDA/HIP have —
//! on wgpu the kernel fails naga validation and the output silently stays zero. With the width
//! in the binding type, the regroup asks for the width the buffer already has and cubecl
//! short-circuits it to a no-op.
//!
//! In-kernel a `VecTensor<E>` *is* a `Tensor<E>` (same expand type): `shape`/`stride` metadata
//! stays scalar-unit, and [`vector_size`](Tensor::vector_size) reads the binding's width.

use core::marker::PhantomData;
use core::ops::Deref;

use cubecl::ir::{Id, Instruction, Metadata, Scope, Value};
use cubecl::prelude::*;
use cubecl::unexpanded;

/// A `Tensor<E>` whose launched binding is typed `Vector<E, v>` (`v` from [`VecTensorArg`]).
/// Only meaningful behind a reference in a launchable struct or kernel signature.
pub struct VecTensor<E: Numeric> {
    _e: PhantomData<E>,
}

/// Unexpanded stand-in only (cube functions never run): lets `#[cube]` bodies call the
/// `Tensor` surface (`shape`, `stride`, `as_slice`, `vector_size`) on a `VecTensor` param.
impl<E: Numeric> Deref for VecTensor<E> {
    type Target = Tensor<E>;

    fn deref(&self) -> &Tensor<E> {
        unexpanded!()
    }
}

impl<E: Numeric> CubeType for VecTensor<E> {
    type ExpandType = TensorExpand<E>;
}

/// The runtime argument: the plain tensor handle plus the width its binding is typed at.
/// Shape/strides stay scalar-unit; `vector_size > 1` requires a contiguous innermost axis the
/// width divides ([`new`](VecTensorArg::new) asserts it, `Launcher::vector_size` picks it).
pub struct VecTensorArg<R: Runtime> {
    tensor: TensorArg<R>,
    vector_size: usize,
}

impl<R: Runtime> VecTensorArg<R> {
    pub fn new(tensor: TensorArg<R>, vector_size: usize) -> Self {
        if vector_size > 1 {
            assert_eq!(
                tensor.strides().last(),
                Some(&1),
                "VecTensorArg: a wide binding needs a contiguous innermost axis"
            );
            assert!(
                tensor
                    .shape()
                    .last()
                    .is_some_and(|e| e.is_multiple_of(vector_size)),
                "VecTensorArg: vector_size must divide the innermost extent"
            );
        }
        VecTensorArg {
            tensor,
            vector_size,
        }
    }
}

/// [`TensorCompilationArg`] plus the binding width, so `expand` can rebuild the vectorized
/// type and two widths never share a compiled kernel.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VecTensorCompilationArg {
    tensor: TensorCompilationArg,
    vector_size: usize,
}

/// Mirror of `LaunchArg for Tensor<C>`, with the element type widened to the line.
impl<E: Numeric> LaunchArg for VecTensor<E> {
    type RuntimeArg<R: Runtime> = VecTensorArg<R>;
    type CompilationArg = VecTensorCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let ty = launcher
            .with_scope(|scope| E::__expand_as_type(scope))
            .with_vector_size(arg.vector_size);
        // The binding indexes in lines, so its buffer length is a line count.
        let len = arg.tensor.size() / arg.vector_size;
        let meta_arg = TensorMetaLaunch::new(len, arg.tensor.shape().len());
        let buffer = match &arg.tensor {
            TensorArg::Handle { .. } => BufferCompilationArg { inplace: None },
            TensorArg::Alias { input_pos, .. } => BufferCompilationArg {
                inplace: Some(*input_pos as Id),
            },
        };
        launcher.register_tensor(arg.tensor, ty);
        let meta = TensorMeta::register(meta_arg, launcher);
        VecTensorCompilationArg {
            tensor: TensorCompilationArg { meta, buffer },
            vector_size: arg.vector_size,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorExpand<E> {
        let buffer = match arg.tensor.buffer.inplace {
            Some(id) => builder.inplace(id),
            None => {
                let ty = E::__expand_as_type(&builder.scope).with_vector_size(arg.vector_size);
                builder.tensor(ty)
            }
        };
        let meta = TensorMeta::expand(&arg.tensor.meta, builder);
        let scope = &builder.scope;
        let len = buffer_length(scope, buffer);
        let buffer = cubecl::frontend::slice::from_raw_parts::<E>(
            scope,
            buffer,
            0usize.into_expand(scope),
            len.into(),
        );
        TensorExpand::__expand_from_parts(meta, buffer)
    }
}

/// cubecl's `expand_buffer_length_native`, which is crate-private there.
fn buffer_length(scope: &Scope, list: Value) -> Value {
    let out = scope.create_value(usize::__expand_as_type(scope));
    scope.register(Instruction::new(Metadata::BufferLength { list }, out));
    out
}
