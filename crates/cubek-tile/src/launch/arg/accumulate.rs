//! An output several instances accumulate into, as a single launch argument.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};
use cubecl::unexpanded;

use crate::*;

/// An output several instances accumulate into, as a single launch argument.
/// The buffer must arrive holding zero: the atomic only ever adds.
#[derive(CubeType, CubeLaunch)]
pub struct AccumulateArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<Atomic<E>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric> AccumulateArg<'a, E> {
    /// Serve the output as a [`Tile`] that accumulates into it at width `V`.
    pub fn tile<V: Size>(&self, #[comptime] space: Partitioning) -> Tile<E> {
        // An atomic element is scalar, so these strides are already in scalars.
        let geometry = RuntimeGeometry::of_tensor::<Atomic<E>>(
            self.tensor,
            comptime!(self.spec.projection.physical_rank()),
        );
        let sink = atomic_sink::<E, V>(self.tensor);
        // A launch-time `Size` has a value only at expansion.
        let width = V::value();
        GlobalOperand::<E>::sink(
            sink,
            geometry,
            width,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            Write::Accumulate,
        )
        .tile(comptime!(space.levels().to_vec()))
    }
}

/// The erased tensor over an atomic buffer, accumulating at width `N`.
// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
fn atomic_sink<E: Numeric, N: Size>(_values: &Tensor<Atomic<E>>) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod atomic_sink {
    use super::*;

    pub(super) fn expand<E: Numeric, N: Size>(
        _scope: &Scope,
        values: &<Tensor<Atomic<E>> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(AtomicAccumulate::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            _n: PhantomData,
        })
    }
}

/// A write-only backing that accumulates into an `Atomic<E>` buffer.
struct AtomicAccumulate<E: Numeric, N: Size> {
    values: <Tensor<Atomic<E>> as CubeType>::ExpandType,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for AtomicAccumulate<E, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    /// In lines of `N`, as the trait counts them, off a buffer whose own elements are scalar: the
    /// buffer's, not the tensor's, since a pitched buffer holds cells past its shape's product.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.values.__expand_buffer_len_method(scope);
        // Read at expansion: a width the launch defines has no host-side value.
        let width = <N as Size>::__expand_value(scope).__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, width)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        accumulate_line::expand::<E, N>(scope, &self.values, index, value.into());
    }
}

impl<E: Numeric, N: Size> WritesLines<E> for AtomicAccumulate<E, N> {}

/// Accumulate one line into the buffer: `N` scalar adds at the line's own offset.
#[cube]
fn accumulate_line<E: Numeric, N: Size>(
    values: &Tensor<Atomic<E>>,
    index: usize,
    value: Vector<E, N>,
) {
    let base = index * N::value();
    #[unroll]
    for k in 0..N::value() {
        values[base + k].fetch_add(value.extract(k));
    }
}
