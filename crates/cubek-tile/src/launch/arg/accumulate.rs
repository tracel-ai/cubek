//! An output several instances accumulate into, as a single launch argument.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};
use cubecl::unexpanded;

use crate::*;

/// An output several instances accumulate into, as a single launch argument: [`TileArg`]'s twin
/// for a destination whose writes add, not replace. `Atomic<E>` carries no served width, so
/// [`tile`](Self::tile) states it; the drain adds each line's scalars one atomic at a time.
///
/// **The buffer arrives holding the monoid's identity.** A cell here belongs to several instances
/// and none of them may seed it, so the seeding happens once at the launch. Nothing can check it:
/// the destination cannot read.
#[derive(CubeType, CubeLaunch)]
pub struct AccumulateArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<Atomic<E>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric> AccumulateArg<'a, E> {
    /// Serve the output as a [`Tile`] that accumulates into it at width `V`. [`TileArg::tile`]'s
    /// twin, and the same call: the kernel's one `space` projected onto this operand's `spec`
    /// axes.
    pub fn tile<V: Size>(&self, #[comptime] space: Partitioning) -> Tile<E> {
        // The geometry a sink cannot be asked for, taken off the buffer behind it. An atomic
        // element is scalar, so these strides are already in the scalars the layout wants.
        let geometry = RuntimeGeometry::of_tensor::<Atomic<E>>(
            self.tensor,
            comptime!(self.spec.projection.physical_rank()),
        );
        let sink = atomic_sink::<E, V>(self.tensor);
        // Read at expansion, not as a Rust constant: a launch-time `Size` has no `value()`
        // until the kernel is being defined.
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
///
/// A constructor here rather than in cubecl for the reason the backing is here: what a write
/// *means* is this crate's statement, and cubecl's own backings all replace.
// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
fn atomic_sink<E: Numeric, N: Size>(_values: &Tensor<Atomic<E>>) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod atomic_sink {
    use super::*;

    pub fn expand<E: Numeric, N: Size>(
        _scope: &Scope,
        values: &<Tensor<Atomic<E>> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(AtomicAccumulate::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            _n: PhantomData,
        })
    }
}

/// A backing that accumulates into an `Atomic<E>` buffer. Writes and never reads, so it declares
/// [`WritesLines`] alone: a partial that could be read back is one a cube could seed from, which
/// is the race this exists to avoid.
struct AtomicAccumulate<E: Numeric, N: Size> {
    values: <Tensor<Atomic<E>> as CubeType>::ExpandType,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for AtomicAccumulate<E, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    /// In lines of `N`, as the trait counts them, off a buffer whose own elements are scalar.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.values.__expand_len_method(scope);
        let width = N::value().__expand_runtime_method(scope);
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
///
/// Scalar because an atomic is: `Atomic<E>` is one element wide whatever the tile serves its
/// lines at, so the width the walk works in is undone here and nowhere else. The tile above keeps
/// addressing whole lines, which is what keeps this a backing rather than a second drain.
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
