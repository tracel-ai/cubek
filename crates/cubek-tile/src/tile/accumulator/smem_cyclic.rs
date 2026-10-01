//! A shared-memory destination the planes of a cube write exclusively, on the cyclic schedule, and
//! a tile that reads the sum back.
//!
//! [`SmemAccumulation`](super::smem_accumulation::SmemAccumulation)'s writers add into every cell
//! at once through atomics, in whatever order they arrive, into a buffer zeroed first. Here each
//! writer's fragments are dealt into as many chunks as there are writers, by their place in the
//! fragment grid, and the drain runs in as many rounds: in round `r`, writer `w` lands its
//! fragments of chunk `(w + r) mod writers` alone, replacing their cells in round zero and adding
//! into them after, with a plain read and write. Every writer works every round, on cells no other
//! writer touches, a cube barrier between rounds, and drains each fragment once: no atomics, no
//! zeroing, and each cell takes its partials in a fixed order, so the sum is the same bits from run
//! to run.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines, WriteOnly,
    WritesLines,
};
use cubecl::unexpanded;

use crate::*;

/// A cube's shared-memory accumulator over one box, summing in `A` and read back as `T`, that
/// `writers` writers reach exclusively, on the [`Cyclic`](Schedule::Cyclic) schedule.
///
/// A writer drains its partial with [`drain`](SmemCyclicAccumulation::drain) (or
/// [`drain_at`](SmemCyclicAccumulation::drain_at) for a partial over one window), which runs every
/// round and synchronizes the cube after each; once every writer has, [`source`] holds the sum.
/// Every unit of the cube reaches the drain, each plane with the partial and window it holds.
///
/// [`source`]: SmemCyclicAccumulation::source
#[derive(CubeType)]
pub struct SmemCyclicAccumulation<A: Numeric, T: Numeric> {
    values: Shared<[A]>,
    /// The box the accumulator was opened over, which every round's sink is shaped like.
    like: Tile<T>,
    /// Reads the sum back as `T`; valid only once every writer has drained.
    pub source: Tile<T>,
    #[cube(comptime)]
    writers: usize,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A shared-memory accumulator over this tile's box, summing in `A`, that `writers` writers
    /// reach exclusively, one chunk each a round. Nothing is zeroed: every cell's first writer
    /// replaces it.
    pub fn smem_cyclic_accumulation<A: Numeric>(
        &self,
        #[comptime] writers: usize,
    ) -> SmemCyclicAccumulation<A, T> {
        let space = comptime!(self.place.space.clone());
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(A::size().comptime())
        ));
        let values = Shared::<[A]>::new_slice(comptime!(form.cells()));
        // The previous open from this call site may still be read; let it finish first.
        sync_cube();

        let units = self.units();
        let source = Memory::<T>::smem_backed(
            space,
            1usize,
            comptime!(FillUnits::cube(units)),
            Backing::<T>::new_ReadCall(ErasedTensor::<T, ReadOnly>::of_smem_plain_load::<A>(
                &values,
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
        );
        SmemCyclicAccumulation::<A, T> {
            values,
            like: self.clone(),
            source: source.nested_like::<T>(self),
            writers,
        }
    }
}

#[cube]
impl<A: Numeric, T: Numeric> SmemCyclicAccumulation<A, T> {
    /// The sink every writer drains into in `round`: it replaces each cell it is handed in round
    /// zero and adds into it after.
    pub fn sink(&self, round: usize) -> Tile<A> {
        let space = comptime!(self.like.place.space.clone());
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(A::size().comptime())
        ));
        let units = self.like.units();
        let sink = Memory::<A>::smem_backed(
            space,
            1usize,
            comptime!(FillUnits::cube(units)),
            Backing::<A>::new_WriteCall(ErasedTensor::<A, WriteOnly>::of_smem_cyclic(
                &self.values,
                round == 0,
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Exclusive(Schedule::Cyclic)),
        );
        sink.nested_like::<T>(&self.like)
    }

    /// Drain `partial`, an accumulator over the whole box, every round of the schedule, the cube
    /// synchronized after each: each unit lands its plane's fragments of the round's chunk as
    /// `writer`, its plane's place among the writers. Once every writer of the cube has, the sum is
    /// in [`source`](SmemCyclicAccumulation::source).
    pub fn drain<P: Numeric>(&self, partial: &Tile<P>, writer: usize) {
        #[unroll]
        for round in 0..self.writers {
            let turn = (writer + round) % comptime!(self.writers);
            partial.drained_chunk_into(&self.sink(round), turn, comptime!(self.writers));
            sync_cube();
        }
    }

    /// [`drain`](SmemCyclicAccumulation::drain) for a partial over one `window` of the box, as a
    /// kernel that walks its planes itself holds one.
    pub fn drain_at<P: Numeric>(&self, partial: &Tile<P>, window: &Region, writer: usize) {
        #[unroll]
        for round in 0..self.writers {
            let turn = (writer + round) % comptime!(self.writers);
            partial.drained_chunk_into(&self.sink(round).at(window), turn, comptime!(self.writers));
            sync_cube();
        }
    }
}

/// The erased tensors over a shared buffer of plain values.
pub(crate) trait SmemCyclicSink<E: Numeric> {
    /// The sink that writes into `values` one scalar a line, replacing each cell where `first`
    /// and adding into it otherwise.
    fn of_smem_cyclic(_values: &Shared<[E]>, _first: bool) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_cyclic(
        _scope: &Scope,
        values: &<Shared<[E]> as CubeType>::ExpandType,
        first: NativeExpand<bool>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(SmemCyclic::<E> {
            values: ExpandTypeClone::clone_unchecked(values),
            first,
            _e: PhantomData,
        })
    }
}

impl<E: Numeric> SmemCyclicSink<E> for ErasedTensor<E, WriteOnly> {}

/// The source that loads out of a shared buffer of plain `A`, served as `T`.
pub(crate) trait SmemPlainLoadSource<T: Numeric> {
    /// The source that loads `values`, cast to `T`, one scalar per line.
    fn of_smem_plain_load<A: Numeric>(_values: &Shared<[A]>) -> ErasedTensor<T, ReadOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_plain_load<A: Numeric>(
        _scope: &Scope,
        values: &<Shared<[A]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<T, ReadOnly> {
        ErasedTensorExpand::new(SmemPlainLoad::<A, T> {
            values: ExpandTypeClone::clone_unchecked(values),
            _t: PhantomData,
        })
    }
}

impl<T: Numeric> SmemPlainLoadSource<T> for ErasedTensor<T, ReadOnly> {}

/// A backing that replaces each cell it is handed, or adds into it, in a shared buffer: a read and
/// a write, not an atomic, so it declares [`WritesLines`] alone. The rounds keep one writer at a
/// cell at a time.
struct SmemCyclic<E: Numeric> {
    values: <Shared<[E]> as CubeType>::ExpandType,
    first: NativeExpand<bool>,
    _e: PhantomData<E>,
}

impl<E: Numeric> ErasedTensorOperationsExpand<E> for SmemCyclic<E> {
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
        cyclic_cell::expand::<E>(scope, &mut self.values, index, value.into(), self.first);
    }
}

impl<E: Numeric> WritesLines<E> for SmemCyclic<E> {}

/// A backing that loads out of a shared buffer of plain `A`, casting to `T`.
struct SmemPlainLoad<A: Numeric, T: Numeric> {
    values: <Shared<[A]> as CubeType>::ExpandType,
    _t: PhantomData<T>,
}

impl<A: Numeric, T: Numeric> ErasedTensorOperationsExpand<T> for SmemPlainLoad<A, T> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        load_plain_cell::expand::<A, T>(scope, &self.values, index).expand
    }
}

impl<A: Numeric, T: Numeric> ReadsLines<T> for SmemPlainLoad<A, T> {}

/// Land one line, one scalar wide: the value where `first`, the cell plus the value otherwise.
#[cube]
fn cyclic_cell<E: Numeric>(
    values: &mut Shared<[E]>,
    index: usize,
    value: Vector<E, Const<1>>,
    first: bool,
) {
    if first {
        values[index] = value.extract(0usize);
    } else {
        values[index] += value.extract(0usize);
    }
}

/// Load one cell as a line one scalar wide, cast to `T`.
#[cube]
fn load_plain_cell<A: Numeric, T: Numeric>(
    values: &Shared<[A]>,
    index: usize,
) -> Vector<T, Const<1>> {
    Vector::new(T::cast_from(values[index]))
}
