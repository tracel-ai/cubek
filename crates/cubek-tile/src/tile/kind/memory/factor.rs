//! A factor these values carry ([`Tile::mul`](crate::Tile::mul)): scale tiles windowed with the
//! values and multiplied in where a kernel copies or contracts them. Levels are innermost first.

use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubecl::unexpanded;

use crate::*;

/// One level of a factor: the erased scale tile and its comptime facts.
#[derive(Clone)]
pub(crate) struct FactorLevel {
    read: Arc<dyn FactorRead>,
    /// The axes the scales span, each one of the values'.
    pub(crate) space: Space,
    /// How the scales address their buffer.
    pub(crate) projection: Projection,
    /// Whether reaching this scale is a plane shuffle, which the whole plane takes part in.
    pub(crate) by_shuffle: bool,
}

/// A scale tile as the values read it.
pub(crate) trait FactorRead {
    /// The scale covering the value at `coords` of a tile spanning `values`, widened to `f32`.
    fn scale_for(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: &Space,
    ) -> NativeExpand<f32>;

    /// This scale tile windowed to `step`, as the values it rides are.
    fn at_step(&self, scope: &Scope, step: &StepExpand) -> Arc<dyn FactorRead>;
}

impl<S: Numeric> FactorRead for TileExpand<S> {
    fn scale_for(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: &Space,
    ) -> NativeExpand<f32> {
        let scale = self
            .clone()
            .__expand_scale_for_method(scope, coords, values.clone());
        f32::__expand_cast_from(scope, scale)
    }

    fn at_step(&self, scope: &Scope, step: &StepExpand) -> Arc<dyn FactorRead> {
        Arc::new(self.clone().__expand_at_step_method(scope, step))
    }
}

/// The scales an operand's values carry, innermost first; empty when none.
#[derive(Clone)]
pub(crate) struct Factor;

#[derive(Clone, Default)]
pub(crate) struct FactorExpand {
    pub(crate) levels: Vec<FactorLevel>,
}

impl Factor {
    /// No scales: what every operand carries until [`Tile::mul`](crate::Tile::mul).
    pub(crate) fn none() -> Factor {
        unexpanded!()
    }

    /// This factor windowed to `step`, as the values it rides are.
    pub(crate) fn at(&self, _step: &Step) -> Factor {
        unexpanded!()
    }

    /// Whether the innermost level's scales vary along `axis`, so a run along it spans scales.
    pub(crate) fn varies_along(&self, _axis: Axis) -> bool {
        unexpanded!()
    }

    /// The scale covering the value at `coords`: the product of every level.
    pub(crate) fn at_coords(&self, _coords: &Coords<u32>, _values: Space) -> f32 {
        unexpanded!()
    }

    pub(crate) fn __expand_none(_scope: &Scope) -> FactorExpand {
        FactorExpand::default()
    }
}

impl FactorExpand {
    /// A factor of one level: `scale`, read at the values' coordinates.
    pub(crate) fn of<S: Numeric>(scope: &Scope, scale: &TileExpand<S>) -> Self {
        FactorExpand {
            levels: vec![FactorLevel {
                space: scale.place.space.clone(),
                projection: scale.clone().__expand_projection_method(scope),
                by_shuffle: scale.clone().__expand_by_shuffle_method(scope),
                read: Arc::new(scale.clone()),
            }],
        }
    }

    /// This factor with `coarser`'s levels after its own.
    pub(crate) fn and(&self, coarser: FactorExpand) -> FactorExpand {
        let mut levels = self.levels.clone();
        levels.extend(coarser.levels);
        FactorExpand { levels }
    }

    /// The innermost level, the one a leaf reads per line.
    pub(crate) fn inner(&self) -> Option<&FactorLevel> {
        self.levels.first()
    }

    /// Whether these values carry any scales at all.
    pub(crate) fn scaled(&self) -> bool {
        !self.levels.is_empty()
    }

    /// Whether any level is reached by a plane shuffle, which the whole plane takes part in.
    pub(crate) fn by_shuffle(&self) -> bool {
        self.levels.iter().any(|level| level.by_shuffle)
    }

    /// This factor's innermost level alone.
    pub(crate) fn innermost(&self) -> FactorExpand {
        FactorExpand {
            levels: self.levels.iter().take(1).cloned().collect(),
        }
    }

    /// Every level coarser than the innermost, folded into one value read at the region's origin.
    pub(crate) fn coarse(&self, scope: &Scope) -> NativeExpand<f32> {
        let mut folded: NativeExpand<f32> =
            ExpandValue::constant(1u64.into(), f32::elem_type(scope)).into();
        for level in self.levels.iter().skip(1) {
            let origin = origin_of(scope, &level.space);
            let one = level.read.scale_for(scope, &origin, &level.space);
            folded = MulExpand::__expand_mul_method(folded, scope, one);
        }
        folded
    }

    pub(crate) fn __expand_at_method(&self, scope: &Scope, step: &StepExpand) -> FactorExpand {
        FactorExpand {
            levels: self
                .levels
                .iter()
                .map(|level| FactorLevel {
                    read: level.read.at_step(scope, step),
                    space: level.space.clone(),
                    projection: level.projection.clone(),
                    by_shuffle: level.by_shuffle,
                })
                .collect(),
        }
    }

    pub(crate) fn __expand_at_coords_method(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: Space,
    ) -> NativeExpand<f32> {
        let mut product: Option<NativeExpand<f32>> = None;
        for level in &self.levels {
            let one = level.read.scale_for(scope, coords, &values);
            product = Some(match product {
                None => one,
                Some(product) => MulExpand::__expand_mul_method(product, scope, one),
            });
        }
        product.unwrap_or_else(|| ExpandValue::constant(1u64.into(), f32::elem_type(scope)).into())
    }
}

/// The all-zero coordinate of `space`: where a level covering a whole region is read.
fn origin_of(scope: &Scope, space: &Space) -> CoordsExpand<u32> {
    let mut origin = Coords::<u32>::__expand_new(scope);
    for _ in 0..space.rank() {
        let zero: NativeExpand<u32> =
            ExpandValue::constant(0u64.into(), u32::elem_type(scope)).into();
        origin.__expand_push_method(scope, zero);
    }
    origin
}

/// A factor as a leaf reads it: the innermost level per line, coarser levels as one value.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct FactorReader {
    /// The innermost level, or nothing at all.
    pub(crate) inner: Factor,
    /// Every coarser level, already met.
    pub(crate) coarse: f32,
    /// The values' space, which a line's `(row, col)` resolves to a coordinate through.
    #[cube(comptime)]
    pub(crate) values: Space,
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Whether there is a scale at all.
    #[cube(comptime)]
    pub(crate) scaled: bool,
    /// Which batch matrix the lines are read from.
    pub(crate) matrix: usize,
}

impl FactorReader {
    /// Whether a scale read is a plane shuffle; a reader keeps its units converged around one.
    pub(crate) fn by_shuffle(&self) -> bool {
        unexpanded!()
    }
}

#[cube]
impl FactorReader {
    /// `value`, the line at `pos` of the values' matrix, under the scale covering it.
    pub(crate) fn apply<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        pos: Coords2d,
    ) -> Vector<E, V> {
        if comptime!(self.scaled) {
            let (row, col) = pos;
            let coords = matrix_coords(
                row,
                col,
                self.matrix,
                comptime!(&self.values),
                comptime!(self.axes),
                comptime!(self.vector_size),
            );
            self.apply_at::<E, V>(value, &coords)
        } else {
            value
        }
    }

    /// `value`, the line whose first value lies at `coords`, under the scale covering it.
    pub(crate) fn apply_at<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        coords: &Coords<u32>,
    ) -> Vector<E, V> {
        if comptime!(self.scaled) {
            let scale = self.inner.at_coords(coords, comptime!(self.values.clone()));
            value * Vector::<E, V>::cast_from(scale * self.coarse)
        } else {
            value
        }
    }
}

impl FactorExpand {
    pub(crate) fn __expand_varies_along_method(&self, _scope: &Scope, axis: Axis) -> bool {
        self.inner()
            .is_some_and(|level| level.projection.addresses(axis))
    }
}

impl FactorReaderExpand {
    pub(crate) fn __expand_by_shuffle_method(&self, _scope: &Scope) -> bool {
        self.inner.by_shuffle()
    }
}

impl CubeType for Factor {
    type ExpandType = FactorExpand;
}
impl IntoExpand for FactorExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for FactorExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for FactorExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for FactorExpand {}
impl AsRefExpand for FactorExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for FactorExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
