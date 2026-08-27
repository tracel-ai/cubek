//! The scaled read: one operand's values against another operand's scales.
//!
//! A scale is one more factor of the term, so folding it in belongs where the value is *read*, not
//! in the loop that consumes it. Reading through this, a contraction over a scaled operand is the
//! plain contraction: nothing below the view knows a scale exists, and the leaf keeps one body.
//!
//! The scales answer the same coordinate the values do because both operands span the one
//! [`Space`](crate::Space). Where they vary is their own projection's business: an axis they omit
//! they cannot vary over, which is how one scale covers a whole block with nothing dividing
//! anything.

use cubecl::ir::VectorSize;
use cubecl::prelude::barrier::Barrier;
use cubecl::unexpanded;
use cubecl::{
    prelude::*,
    std::tensor::{View, ViewExpand, ViewOperations, ViewOperationsExpand, layout::Coords2d},
};

/// The scales' coordinate for a values' one.
///
/// The two count different things along their shared edge: the values' column counts lines of
/// `width` values, the scales' counts *blocks*. So the values' column widens to values and then
/// narrows to blocks, both by comptime factors. A line never straddles a block — the block is an
/// axis and the values partition it — so the division is exact.
#[cube]
fn scale_pos(pos: Coords2d, #[comptime] width: usize, #[comptime] per_scale: usize) -> Coords2d {
    let (row, col) = pos;
    let lines_per_scale = comptime!(per_scale / width);
    (row, col / comptime!(lines_per_scale as u32))
}

/// One value line against its scale: the scale is one value however wide the line is, so it
/// broadcasts across the line's lanes.
#[cube]
fn scale_line<E: Numeric, V: Size, S: Numeric, SW: Size>(
    value: Vector<E, V>,
    scale: Vector<S, SW>,
) -> Vector<E, V> {
    value * Vector::<E, V>::cast_from(scale.extract(0usize))
}

/// A [`View`] serving `values ⊗ scales`: each read takes the value at `pos` and the scale at the
/// same `pos`, and multiplies. The scale is one value however wide the line is, so it broadcasts
/// across the line's lanes.
///
/// The scales' element is whatever their tensor holds, because it is a tensor: `S` rides here and
/// meets the value's element in the multiply, with nothing in between to disagree with.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub struct ScaledView<'a, E: Numeric, V: Size, S: Numeric, SW: Size> {
    values: View<'a, Vector<E, V>, Coords2d>,
    scales: View<'a, Vector<S, SW>, Coords2d>,
    /// The values' line width, which is what separates the two column counts.
    #[cube(comptime)]
    width: usize,
    /// Values one scale covers along the shared edge.
    #[cube(comptime)]
    per_scale: usize,
}

#[cube]
impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ScaledView<'a, E, V, S, SW> {
    pub fn new(
        values: View<'a, Vector<E, V>, Coords2d>,
        scales: View<'a, Vector<S, SW>, Coords2d>,
        #[comptime] width: usize,
        #[comptime] per_scale: usize,
    ) -> Self {
        comptime!(assert!(
            per_scale.is_multiple_of(width),
            "ScaledView: one scale covers {per_scale} values of an operand served {width} at a \
             time, so a line of it straddles two scales"
        ));
        ScaledView::<'a, E, V, S, SW> {
            values,
            scales,
            width,
            per_scale,
        }
    }
}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ScaledView<'a, E, V, S, SW> {
    /// This view as a plain [`View`] of scaled values, which is what every reader takes.
    pub fn view(self) -> View<'a, Vector<E, V>, Coords2d> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &Scope,
        this: ScaledViewExpand<'a, E, V, S, SW>,
    ) -> ViewExpand<'a, Vector<E, V>, Coords2d> {
        this.__expand_view_method(scope)
    }
}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ScaledViewExpand<'a, E, V, S, SW> {
    fn at_scale(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> <Coords2d as CubeType>::ExpandType {
        scale_pos::expand(scope, pos, self.width, self.per_scale)
    }

    fn scale(
        &self,
        scope: &Scope,
        value: NativeExpand<Vector<E, V>>,
        scale: NativeExpand<Vector<S, SW>>,
    ) -> NativeExpand<Vector<E, V>> {
        scale_line::expand::<E, V, S, SW>(scope, value, scale)
    }

    pub fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, Vector<E, V>, Coords2d> {
        ViewExpand::new(scope, self)
    }
}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> Vectorized for ScaledView<'a, E, V, S, SW> {}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> VectorizedExpand
    for ScaledViewExpand<'a, E, V, S, SW>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.values.__expand_vector_size_method(scope)
    }
}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ViewOperations<Vector<E, V>, Coords2d>
    for ScaledView<'a, E, V, S, SW>
{
}

impl<'a, E: Numeric, V: Size, S: Numeric, SW: Size> ViewOperationsExpand<Vector<E, V>, Coords2d>
    for ScaledViewExpand<'a, E, V, S, SW>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let value = self.values.clone().__expand_read_method(scope, pos);
        let scale = self
            .scales
            .clone()
            .__expand_read_method(scope, self.at_scale(scope, pos));
        self.scale(scope, value, scale)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let value = self.values.clone().__expand_read_checked_method(scope, pos);
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, self.at_scale(scope, pos));
        self.scale(scope, value, scale)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
        mask_value: NativeExpand<Vector<E, V>>,
    ) -> NativeExpand<Vector<E, V>> {
        let value = self.values.clone().__expand_read_checked_method(scope, pos);
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, self.at_scale(scope, pos));
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let scaled = self.scale(scope, value, scale);
        select::expand::<Vector<E, V>>(scope, in_bounds, scaled, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<E, V>> {
        let value = self
            .values
            .clone()
            .__expand_read_unchecked_method(scope, pos);
        let scale = self
            .scales
            .clone()
            .__expand_read_unchecked_method(scope, self.at_scale(scope, pos));
        self.scale(scope, value, scale)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <Coords2d as CubeType>::ExpandType,
        _end: <Coords2d as CubeType>::ExpandType,
    ) -> &SliceExpand<Vector<E, V>> {
        panic!("ScaledView: a scaled operand has no raw slice of served values")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <Coords2d as CubeType>::ExpandType {
        self.values.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: <Coords2d as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        self.values.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<E, V>>,
        _pos: <Coords2d as CubeType>::ExpandType,
    ) {
        panic!("ScaledView: a tensor map cannot fold a scale on the fly")
    }
}
