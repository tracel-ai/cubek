//! One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};
use cubecl::unexpanded;

use crate::kind::Write;
use crate::*;

/// One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric, V: Size> {
    pub tensor: &'a Tensor<Vector<E, V>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`] over the kernel's one `space`.
    pub fn tile(&self, #[comptime] space: Partitioning) -> Tile<E> {
        GlobalOperand::<E>::tensor(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) for a destination written with [`Write::Fold`]: each line the drain
    /// writes is added to the line already there, read back and written whole, rather than
    /// replacing it. The kernel gives the instances sharing a cell their turns; this only adds.
    pub fn folded_tile(&self, #[comptime] space: Partitioning) -> Tile<E> {
        let geometry = RuntimeGeometry::of_tensor::<Vector<E, V>>(
            self.tensor,
            comptime!(self.spec.projection.physical_rank()),
        );
        GlobalOperand::<E>::sink(
            fold_sink::<E, V>(self.tensor),
            geometry,
            V::value(),
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            Write::Fold,
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) reading `O` out of the stored `E`, unpacked where the binding is
    /// [`packed`](TileSpec::packed).
    pub fn tile_as<O: Numeric>(&self, #[comptime] space: Partitioning) -> Tile<O> {
        GlobalOperand::<O>::stored(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) for a partly runtime gather map, its dynamic values in
    /// [`GlobalOperand::gathered`]'s order.
    pub fn tile_gathered(
        &self,
        #[comptime] space: Partitioning,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<E> {
        GlobalOperand::<E>::gathered(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            coefficients,
            offsets,
        )
        .tile(comptime!(space.levels().to_vec()))
    }
}

/// The erased tensor over a plain buffer that folds each line written into the one already there.
///
/// A constructor here rather than in cubecl for the reason the atomic sink's is: what a write
/// *means* is this crate's statement, and cubecl's own backings all replace.
// `N` is read by the expansion, which is where a `Size` has a value.
#[allow(clippy::extra_unused_type_parameters)]
fn fold_sink<E: Numeric, N: Size>(_values: &Tensor<Vector<E, N>>) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod fold_sink {
    use super::*;

    pub fn expand<E: Numeric, N: Size>(
        _scope: &Scope,
        values: &<Tensor<Vector<E, N>> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(FoldAccumulate::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            _n: PhantomData,
        })
    }
}

/// A backing that adds each line written into the line it lands on: a read and a write, not an
/// atomic, so it declares [`WritesLines`] alone. The read is the write's own, never the drain's:
/// no cube seeds from a cell another is folding into.
struct FoldAccumulate<E: Numeric, N: Size> {
    values: <Tensor<Vector<E, N>> as CubeType>::ExpandType,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for FoldAccumulate<E, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    /// The buffer's lines, not the tensor's: a pitched buffer holds lines past its shape's
    /// product, and the rows a layout addresses there are written like any other.
    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_buffer_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        fold_line::expand::<E, N>(scope, &mut self.values, index, value.into());
    }
}

impl<E: Numeric, N: Size> WritesLines<E> for FoldAccumulate<E, N> {}

/// Fold one line into the buffer: the line there plus `value`, written whole.
#[cube]
fn fold_line<E: Numeric, N: Size>(
    values: &mut Tensor<Vector<E, N>>,
    index: usize,
    value: Vector<E, N>,
) {
    values[index] += value;
}
