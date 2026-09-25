//! The destination a contraction cut across a cube's planes drains into: a shared-memory buffer
//! whose writes add into the cell atomically, and a tile that reads the sum back.
//!
//! [`AccumulateArg`](crate::AccumulateArg) is the same destination one scope up, in global memory.
//!
//! What the caller owns is the order. The buffer is zeroed and the cube synchronized before
//! [`smem_accumulation`](Tile::smem_accumulation) returns, so the first add lands on the identity.
//! Reading the source before every plane has drained and the cube has synchronized again reads a
//! partial sum, and nothing here can tell.
//!
//! A call site's buffer is the same memory every time it runs, so an accumulator opened in a loop
//! reuses the last one's cells. The cube synchronizes before zeroing too, so a unit still reading
//! the previous sum finishes before the next one clears it.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines, WriteOnly,
    WritesLines,
};
use cubecl::unexpanded;

use crate::*;

/// A cube's shared-memory accumulator over one box: where the planes add their partials, summed
/// in `A`, and where the sum is read back in the box's own element `T`.
#[derive(CubeType)]
pub struct SmemAccumulation<A: Numeric, T: Numeric> {
    /// Writes add into the cell atomically: what each plane's partial drains into.
    pub sink: Tile<A>,
    /// Reads the sum back, cast to `T`: valid once every plane has drained and the cube has
    /// synchronized.
    pub source: Tile<T>,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A shared-memory accumulator over this tile's box, summing in `A`, zeroed, with the cube
    /// synchronized on both sides of the zeroing.
    ///
    /// Sized to this tile's own space, so a cube opens one over its box of the output and each of
    /// its planes drains into `sink.at(&plane)`; the source reads the sum as this tile's element,
    /// so it copies into the box as it stands. Scalar, because an atomic is: whatever width the
    /// planes compute in, a line is added one element at a time.
    pub fn smem_accumulation<A: Numeric>(&self) -> SmemAccumulation<A, T> {
        let space = comptime!(self.place.space.clone());
        let elem_bytes = A::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(elem_bytes)
        ));
        let cells = comptime!(form.cells());
        let values = Shared::<[Atomic<A>]>::new_slice(cells);
        // The previous open from this call site may still be read; let it finish first.
        sync_cube();
        let mut cell = UNIT_POS as usize;
        while cell < cells {
            values[cell].store(A::from_int(0));
            cell += CUBE_DIM as usize;
        }
        sync_cube();

        let units = self.units();
        let sink = Memory::<A>::smem_backed(
            comptime!(space.clone()),
            1usize,
            units,
            Backing::<A>::new_WriteCall(ErasedTensor::<A, WriteOnly>::of_smem_accumulate(&values)),
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Accumulate),
        );
        let source = Memory::<T>::smem_backed(
            space,
            1usize,
            units,
            Backing::<T>::new_ReadCall(ErasedTensor::<T, ReadOnly>::of_smem_load::<A>(&values)),
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
        );
        SmemAccumulation::<A, T> {
            sink: sink.nested_like::<T>(self),
            source: source.nested_like::<T>(self),
        }
    }
}

/// The erased tensors over a shared buffer of atomics: one adds into it, one loads out of it.
///
/// Constructors here rather than in cubecl for the reason [`AccumulateArg`](crate::AccumulateArg)'s is: what a
/// write *means* is this crate's statement.
pub(crate) trait SmemAccumulateSink<E: Numeric> {
    /// The sink that adds into `values`, one scalar per line.
    fn of_smem_accumulate(_values: &Shared<[Atomic<E>]>) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_accumulate(
        _scope: &Scope,
        values: &<Shared<[Atomic<E>]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(SmemAccumulate::<E> {
            values: ExpandTypeClone::clone_unchecked(values),
            _e: PhantomData,
        })
    }
}

impl<E: Numeric> SmemAccumulateSink<E> for ErasedTensor<E, WriteOnly> {}

/// The source that loads out of a buffer of atomics summing in `A`, served as `T`.
pub(crate) trait SmemLoadSource<T: Numeric> {
    /// The source that loads `values`, cast to `T`, one scalar per line.
    fn of_smem_load<A: Numeric>(_values: &Shared<[Atomic<A>]>) -> ErasedTensor<T, ReadOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_load<A: Numeric>(
        _scope: &Scope,
        values: &<Shared<[Atomic<A>]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<T, ReadOnly> {
        ErasedTensorExpand::new(SmemLoad::<A, T> {
            values: ExpandTypeClone::clone_unchecked(values),
            _t: PhantomData,
        })
    }
}

impl<T: Numeric> SmemLoadSource<T> for ErasedTensor<T, ReadOnly> {}

/// A backing that adds into a shared buffer of `Atomic<E>`. Writes and never reads.
struct SmemAccumulate<E: Numeric> {
    values: <Shared<[Atomic<E>]> as CubeType>::ExpandType,
    _e: PhantomData<E>,
}

impl<E: Numeric> ErasedTensorOperationsExpand<E> for SmemAccumulate<E> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        accumulate_cell::expand::<E>(scope, &self.values, index, value.into());
    }
}

impl<E: Numeric> WritesLines<E> for SmemAccumulate<E> {}

/// A backing that loads out of a shared buffer of `Atomic<A>`, casting to `T`. Reads and never
/// writes.
struct SmemLoad<A: Numeric, T: Numeric> {
    values: <Shared<[Atomic<A>]> as CubeType>::ExpandType,
    _t: PhantomData<T>,
}

impl<A: Numeric, T: Numeric> ErasedTensorOperationsExpand<T> for SmemLoad<A, T> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        load_cell::expand::<A, T>(scope, &self.values, index).expand
    }
}

impl<A: Numeric, T: Numeric> ReadsLines<T> for SmemLoad<A, T> {}

/// Add one line, one scalar wide, into its cell.
#[cube]
fn accumulate_cell<E: Numeric>(
    values: &Shared<[Atomic<E>]>,
    index: usize,
    value: Vector<E, Const<1>>,
) {
    values[index].fetch_add(value.extract(0usize));
}

/// Load one cell as a line one scalar wide, cast to `T`.
#[cube]
fn load_cell<A: Numeric, T: Numeric>(
    values: &Shared<[Atomic<A>]>,
    index: usize,
) -> Vector<T, Const<1>> {
    Vector::new(T::cast_from(values[index].load()))
}
