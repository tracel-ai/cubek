//! The destination a contraction cut across cubes drains into: a store whose writes *fold* into
//! the cell rather than replace it.
//!
//! Cutting a contraction at cube scope leaves every cube holding a slice of each output cell it
//! touches. Adding the slices up is the only thing that is missing, and this is the way that costs
//! no second pass: each cube folds its own slice into the cell atomically and never learns that
//! the others exist.
//!
//! It rides the machinery a fused epilogue already uses. A sink is
//! "[a destination] written through its layout and never read", which is exactly what a partial
//! is, so the walk, the layout, the masking and the drain are the ones a plain store gets, and
//! only the last step differs: [`ErasedTensor`] ends the walk in a call, and this backing's call
//! is [`Atomic::fetch_add`] rather than an assignment.
//!
//! Two things the caller owns, neither checkable here. The buffer holds the fold's identity before
//! the launch, since the first cube to arrive folds onto what is there. And the order the folds
//! land in is the order the cubes run in, so the sum is not bit-identical run to run: a launch
//! that needs reproducibility wants the split spelled as an axis instead (`tests/tile/split_k.rs`).

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::unexpanded;
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};

use crate::*;

/// Fold one line into the buffer: `N` scalar folds at the line's own offset.
///
/// Scalar because an atomic is: `Atomic<E>` is one element wide whatever the tile serves its
/// lines at, so the width the walk works in is undone here and nowhere else. The tile above keeps
/// addressing whole lines, which is what keeps this a backing rather than a second drain.
#[cube]
fn fold_line<E: Numeric, N: Size>(values: &Tensor<Atomic<E>>, index: usize, value: Vector<E, N>) {
    let base = index * N::value();
    #[unroll]
    for k in 0..N::value() {
        values[base + k].fetch_add(value.extract(k));
    }
}

/// The backing's length in lines of `N`, off a buffer counted in scalars.
#[cube]
fn lines_of(scalars: usize, #[comptime] width: usize) -> usize {
    scalars / width.runtime()
}

/// A backing that folds into an `Atomic<E>` buffer. Writes and never reads, so it declares
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
        lines_of::expand(scope, scalars, N::value())
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        fold_line::expand::<E, N>(scope, &self.values, index, value.into());
    }
}

impl<E: Numeric, N: Size> WritesLines<E> for AtomicAccumulate<E, N> {}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A tile whose writes fold into `values` instead of replacing, at served width `N`.
    ///
    /// Reached through [`FoldArg`], which is where a kernel names `N` once; this is the
    /// constructor behind it. See [`Tile::of_folding_sink`] for what the caller owns.
    pub fn folding<N: Size>(
        values: &Tensor<Atomic<T>>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        // The geometry a sink cannot be asked for, taken off the buffer behind it. An atomic
        // element is scalar, so these strides are already in the scalars the layout wants.
        let geometry =
            RuntimeGeometry::of_tensor::<Atomic<T>>(values, comptime!(spec.projection.physical_rank()));
        let sink = ErasedTensor::<T, WriteOnly>::of_atomic_accumulate::<N>(values);
        Tile::<T>::of_folding_sink(sink, geometry, comptime!(N::value()), space, spec)
    }
}

/// The erased tensor over an atomic buffer, folding at width `N`.
///
/// A constructor here rather than in cubecl for the reason the backing is here: what a write
/// *means* is this crate's statement, and cubecl's own backings all replace.
pub trait AtomicAccumulateSink<E: Numeric> {
    /// The sink that folds `values` at width `N`.
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
