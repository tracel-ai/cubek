//! The destination a contraction cut across cubes drains into: a store whose writes *accumulate*
//! into the cell rather than replace it.
//!
//! Cutting a contraction at cube scope leaves every cube holding a slice of each output cell it
//! touches. Adding the slices up is the only thing that is missing, and this is the way that costs
//! no second pass: each cube adds its own slice into the cell atomically and never learns that
//! the others exist.
//!
//! It rides the machinery a fused epilogue already uses. A sink is
//! "[a destination] written through its layout and never read", which is exactly what a partial
//! is, so the walk, the layout, the masking and the drain are the ones a plain store gets, and
//! only the last step differs: [`ErasedTensor`] ends the walk in a call, and this backing's call
//! is [`Atomic::fetch_add`] rather than an assignment.
//!
//! Two things the caller owns, neither checkable here. The buffer holds the monoid's identity
//! before the launch, since the first cube to arrive adds onto what is there. And the order the
//! adds land in is the order the cubes run in, so the sum is not bit-identical run to run; a
//! launch that needs reproducibility gives the output an axis for the split instead, which makes
//! every instance's slice a cell of its own and needs no combine at all.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};
use cubecl::unexpanded;

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// A tile whose writes accumulate into `values` instead of replacing, at served width `N`.
    ///
    /// The constructor behind [`AccumulateArg`], which is the surface a kernel binds. See
    /// [`Tile::of_sink`] for what the caller owns.
    pub(crate) fn of_atomic_accumulate<N: Size>(
        values: &Tensor<Atomic<T>>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        // The geometry a sink cannot be asked for, taken off the buffer behind it. An atomic
        // element is scalar, so these strides are already in the scalars the layout wants.
        let geometry = RuntimeGeometry::of_tensor::<Atomic<T>>(
            values,
            comptime!(spec.projection.physical_rank()),
        );
        let sink = ErasedTensor::<T, WriteOnly>::of_atomic_accumulate::<N>(values);
        Tile::<T>::of_sink(
            sink,
            geometry,
            comptime!(N::value()),
            space,
            spec,
            Write::Accumulate,
        )
    }
}

/// The erased tensor over an atomic buffer, accumulating at width `N`.
///
/// A constructor here rather than in cubecl for the reason the backing is here: what a write
/// *means* is this crate's statement, and cubecl's own backings all replace. Reached only through
/// [`Tile::of_atomic_accumulate`], which is why it is not part of the crate's surface.
pub(crate) trait AtomicAccumulateSink<E: Numeric> {
    /// The sink that accumulates into `values` at width `N`.
    fn of_atomic_accumulate<N: Size>(_values: &Tensor<Atomic<E>>) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_atomic_accumulate<N: Size>(
        _scope: &Scope,
        values: &<Tensor<Atomic<E>> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(AtomicAccumulate::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            _n: PhantomData,
        })
    }
}

impl<E: Numeric> AtomicAccumulateSink<E> for ErasedTensor<E, WriteOnly> {}

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
