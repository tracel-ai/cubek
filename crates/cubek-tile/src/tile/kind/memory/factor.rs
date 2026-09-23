//! A factor these values carry: another tile, read at their own coordinates, multiplying them
//! where they are read.
//!
//! A quantized tensor is values *and* scales, and a binding names one thing, so the scales arrive
//! as an operand of their own. [`Tile::mul`](crate::Tile::mul) hands them to the values, and from
//! there they ride along: windowed by the same `at`, read at the same coordinates, and multiplied
//! in at the one place that can do it without materializing a dequantized tile, the read.
//!
//! **The scales' element is the scales' business.** A `Memory<T>` carries a factor whatever the
//! scales are served as: the erasure happens while the kernel is expanded, so nothing downstream
//! of `mul` is generic over it and nothing here is a virtual call in the kernel. A scale arrives
//! as `f32` from the erasure, which is what every scale element widens to without loss.
//!
//! A factor may carry several levels, innermost first. The innermost is read per line, at the
//! coordinate of the line's first value; every coarser one covers the whole region a leaf runs
//! over, so [`FactorReader`] reads it once, at the origin, which is what makes depth cheap.

use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords2d;
use cubecl::unexpanded;

use crate::*;

/// One level of a factor, as a leaf meets it: the scale tile erased behind its two reads, and the
/// comptime facts the statement is checked by.
#[derive(Clone)]
pub(crate) struct FactorLevel {
    read: Arc<dyn FactorRead>,
    /// The axes the scales span. Every one is the values': a scale is looked up at the value's own
    /// coordinate, and an axis one scale holds whole is omitted rather than divided.
    pub(crate) space: Space,
    /// How the scales address their buffer, which says which of their axes they resolve.
    pub(crate) projection: Projection,
    /// Whether reaching this scale is a plane shuffle, which the whole plane takes part in.
    pub(crate) by_shuffle: bool,
}

/// A scale tile as the values read it. Implemented for the tile's expand type, which is where the
/// scales' element still exists; a [`FactorLevel`] holds one of these and is generic over nothing.
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

/// The scales an operand's values carry, innermost first. Empty is an operand carrying none,
/// which reads as it lies and emits no arithmetic at all.
///
/// A [`CubeType`] with no runtime fields of its own: what it holds are the erased scale tiles,
/// whose handles are the runtime state, as a procedural source holds its recipe.
#[derive(Clone)]
pub struct Factor;

#[derive(Clone, Default)]
pub struct FactorExpand {
    pub(crate) levels: Vec<FactorLevel>,
}

impl Factor {
    /// No scales: what every operand carries until [`Tile::mul`](crate::Tile::mul) says otherwise.
    pub(crate) fn none() -> Factor {
        unexpanded!()
    }

    /// This factor windowed to `step`, as the values it rides are.
    pub(crate) fn at(&self, _step: &Step) -> Factor {
        unexpanded!()
    }

    /// The scale covering the value at `coords` of a tile spanning `values`: the product of every
    /// level this factor holds, each looked up at its own granularity.
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

    /// Every level coarser than the innermost, folded into one value read at the region's origin:
    /// they cover the whole region, so they have no position of their own inside it.
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

/// A factor as a leaf reads it: the innermost level, looked up at the coordinates of every line,
/// and every coarser level already met and carried as one value.
///
/// Built once where a leaf opens its operand and read per line. A factor carrying nothing reads as
/// the values lie: the multiply is comptime-absent.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct FactorReader {
    /// The innermost level, or nothing at all.
    pub(crate) inner: Factor,
    /// Every coarser level, already met.
    pub(crate) coarse: f32,
    /// The values' space, and the matrix a leaf reads them as: what a line's `(row, col)`
    /// resolves to a coordinate through.
    #[cube(comptime)]
    pub(crate) values: Space,
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Whether there is a scale at all, which decides whether anything is emitted.
    #[cube(comptime)]
    pub(crate) scaled: bool,
    /// Which batch matrix the lines are read from.
    pub(crate) matrix: usize,
}

impl FactorReader {
    /// Whether a read of the scales reaches the lane that asks by a plane shuffle, which the
    /// whole plane takes part in: a reader keeps its lanes converged around one.
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

    /// `value`, the line whose first value lies at `coords` of the values' space, under the scale
    /// covering it.
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
