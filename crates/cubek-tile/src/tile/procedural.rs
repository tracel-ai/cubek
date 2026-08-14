//! A memory-free tile source evaluated from logical coordinates.

use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::Scope;
use cubecl::unexpanded;
use cubecl::{
    prelude::{barrier::Barrier, *},
    std::tensor::{ViewOperations, ViewOperationsExpand, layout::CoordsDyn},
};
use cubecl_common::Ratio;

use crate::{Coords, CoordsExpand, Fold, FoldExpand, Region, Space, StagePlan};

/// An N-dimensional scalar field evaluated at absolute logical coordinates.
#[cube(expand_base_traits = "ExpandTypeClone")]
pub trait Recipe<T: Numeric> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T;
}

/// A constant procedural field.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Constant<T: Numeric> {
    pub value: T,
}

#[cube]
impl<T: Numeric> Recipe<T> for Constant<T> {
    fn evaluate(&self, _coordinates: &Coords<u32>, #[comptime] _space: Space) -> T {
        self.value
    }
}

/// A one-dimensional affine coordinate expression, `offset + coefficient * coordinate[axis]`.
/// The axis is compile-time metadata; offset and coefficient can be runtime values.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct AffineCoordinates<T: Numeric> {
    pub offset: T,
    pub coefficient: T,
    #[cube(comptime)]
    pub axis: crate::Axis,
}

#[cube]
impl<T: Numeric> Recipe<T> for AffineCoordinates<T> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        self.offset
            + self.coefficient * T::cast_from(coordinates.at(comptime!(space.position(self.axis))))
    }
}

/// Triangle filter over the value of an inner recipe, usually an [`AffineCoordinates`].
#[derive(CubeType, Clone)]
pub struct Linear<C: CubeType> {
    pub coordinate: C,
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Linear<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        let x = self.coordinate.evaluate(coordinates, space).abs();
        select(x < T::new(1.0_f32), T::new(1.0_f32) - x, T::new(0.0_f32))
    }
}

/// Keys' cubic-convolution filter over the value of an inner recipe. `a` shapes the kernel;
/// [`catmull_rom`](Self::catmull_rom) and [`sharp`](Self::sharp) pick the two usual values.
#[derive(CubeType, Clone)]
pub struct Cubic<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub a: Ratio,
}

impl<C: CubeType> Cubic<C> {
    /// The interpolating member of the family, `a = -1/2`.
    pub fn catmull_rom(coordinate: C) -> Self {
        Self {
            coordinate,
            a: Ratio::new(-1, 2),
        }
    }

    /// The sharper `a = -3/4` that image resamplers usually pick.
    pub fn sharp(coordinate: C) -> Self {
        Self {
            coordinate,
            a: Ratio::new(-3, 4),
        }
    }
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Cubic<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        let a = T::new(comptime!(self.a.as_f32()));
        let x = self.coordinate.evaluate(coordinates, space).abs();
        let x2 = x * x;
        let x3 = x2 * x;
        let first = (a + T::new(2.0_f32)) * x3 - (a + T::new(3.0_f32)) * x2 + T::new(1.0_f32);
        let second =
            a * x3 - T::new(5.0_f32) * a * x2 + T::new(8.0_f32) * a * x - T::new(4.0_f32) * a;
        select(
            x <= T::new(1.0_f32),
            first,
            select(x <= T::new(2.0_f32), second, T::new(0.0_f32)),
        )
    }
}

/// Windowed-sinc Lanczos filter over the value of an inner recipe, `sinc(x) * sinc(x / lobes)`
/// inside the support and zero outside it.
#[derive(CubeType, Clone)]
pub struct Lanczos<C: CubeType> {
    pub coordinate: C,
    #[cube(comptime)]
    pub lobes: u8,
}

impl<C: CubeType> Lanczos<C> {
    /// Two lobes, a four-tap kernel.
    pub fn lanczos_2(coordinate: C) -> Self {
        Self {
            coordinate,
            lobes: 2,
        }
    }

    /// Three lobes, a six-tap kernel.
    pub fn lanczos_3(coordinate: C) -> Self {
        Self {
            coordinate,
            lobes: 3,
        }
    }
}

#[cube]
impl<T: Float, C: Recipe<T>> Recipe<T> for Lanczos<C> {
    fn evaluate(&self, coordinates: &Coords<u32>, #[comptime] space: Space) -> T {
        // Zero lobes would leave an empty support and divide by zero below. Checked here rather
        // than in a constructor, which a struct literal can bypass. It fires while the kernel
        // expands, so it surfaces on the client's compilation thread, not at the call site.
        comptime!(assert!(self.lobes > 0, "Lanczos: lobes must be non-zero"));
        let x = self.coordinate.evaluate(coordinates, space);
        let abs_x = x.abs();
        let pi_x = T::new(core::f32::consts::PI) * x;
        let lobes = T::cast_from(self.lobes);
        let denominator = (pi_x * pi_x) / lobes;
        // `select` evaluates both arms, so the singularity at x = 0 is divided away rather than
        // branched around.
        let safe_denominator = select(abs_x < T::new(1e-7_f32), T::new(1.0_f32), denominator);
        select(
            abs_x < T::new(1e-7_f32),
            T::new(1.0_f32),
            select(
                abs_x < lobes,
                (pi_x.sin() * (pi_x / lobes).sin()) / safe_denominator,
                T::new(0.0_f32),
            ),
        )
    }
}

trait RecipeOps<T: Numeric> {
    fn evaluate_virtual(
        &self,
        scope: &Scope,
        coordinates: &CoordsExpand<u32>,
        space: Space,
    ) -> NativeExpand<T>;
}

impl<T: Numeric, R: RecipeExpand<T>> RecipeOps<T> for R {
    fn evaluate_virtual(
        &self,
        scope: &Scope,
        coordinates: &CoordsExpand<u32>,
        space: Space,
    ) -> NativeExpand<T> {
        self.__expand_evaluate_method(scope, coordinates, space)
    }
}

/// Expansion-time type erasure for a [`Recipe`]. It is never a GPU virtual call.
#[derive(Clone)]
pub struct VirtualRecipe<T: Numeric>(PhantomData<T>);

#[derive(Clone)]
pub struct VirtualRecipeExpand<T: Numeric> {
    state: Arc<dyn RecipeOps<T>>,
}

impl<T: Numeric> VirtualRecipe<T> {
    pub fn __expand_new<R: Recipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> VirtualRecipeExpand<T> {
        VirtualRecipeExpand {
            state: Arc::new(recipe),
        }
    }

    pub fn evaluate(&self, _coordinates: &Coords<u32>, _space: Space) -> T {
        unexpanded!()
    }
}

impl<T: Numeric> VirtualRecipeExpand<T> {
    pub fn __expand_evaluate_method(
        &self,
        scope: &Scope,
        coordinates: &CoordsExpand<u32>,
        space: Space,
    ) -> NativeExpand<T> {
        self.state.evaluate_virtual(scope, coordinates, space)
    }
}

impl<T: Numeric> CubeType for VirtualRecipe<T> {
    type ExpandType = VirtualRecipeExpand<T>;
}
impl<T: Numeric> IntoExpand for VirtualRecipeExpand<T> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> ExpandTypeClone for VirtualRecipeExpand<T> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl<T: Numeric> IntoMut for VirtualRecipeExpand<T> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl<T: Numeric> CubeDebug for VirtualRecipeExpand<T> {}
impl<T: Numeric> AsRefExpand for VirtualRecipeExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: Numeric> AsMutExpand for VirtualRecipeExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

/// Runtime state of a procedural source. `origin` tracks regions selected by `Tile::at`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct ProceduralData<T: Numeric> {
    origin: Coords<u32>,
    /// The source's static logical extent. Dynamic axes hold `u32::MAX`, deliberately leaving
    /// them unmasked; unlike `space`, this stays in the parent's coordinate system as
    /// [`Tile::at`](crate::Tile::at) descends into nominally sized trailing partial tiles.
    bound: Coords<u32>,
    /// Whether any level can select a partial tile. It stays with the source while its `space`
    /// descends, because the leaf space alone no longer records an ancestor's overhang.
    #[cube(comptime)]
    pub(crate) bounds_check: bool,
    recipe: VirtualRecipe<T>,
    #[cube(comptime)]
    space: Space,
    /// Where this source lives at each level below. Every level is in place today, since
    /// [`Tile::procedural`](crate::Tile::procedural) is the only constructor and a recipe has no
    /// bytes to stage. The plan is still carried because a level that asks for a stage
    /// cooperatively materializes the recipe into it
    /// ([`MemData::fill_procedural`](crate::MemData)), which is how a source with no bytes would
    /// reach a leaf that cannot evaluate one.
    #[cube(comptime)]
    pub(crate) stage: StagePlan,
    #[cube(comptime)]
    _marker: PhantomData<T>,
}

#[cube]
impl<T: Numeric> ProceduralData<T> {
    pub(crate) fn new_virtual(
        #[comptime] space: Space,
        recipe: VirtualRecipe<T>,
        #[comptime] stage: StagePlan,
    ) -> Self {
        let mut origin = Coords::<u32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            origin.push(0u32.runtime());
            let axis = comptime!(space.axis_at(p));
            let extent = comptime!(if space.is_dynamic(axis) {
                u32::MAX.runtime()
            } else {
                space.runtime_extent_at(p) as u32
            });
            bound.push(extent);
        }
        let bounds_check = comptime!(
            space
                .axes()
                .any(|a| !space.is_dynamic(a) && space.overhangs(a))
        );
        ProceduralData::<T> {
            origin,
            bound,
            bounds_check,
            recipe,
            space,
            stage,
            _marker: PhantomData,
        }
    }

    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> Self {
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            let edge = comptime!(space.partitioner().edge(axis) as u32);
            origin.push(self.origin.at(p) + region.coord(axis).fcast::<u32>() * edge);
        }
        ProceduralData::<T> {
            origin,
            bound: self.bound.clone(),
            bounds_check: comptime!(self.bounds_check),
            recipe: self.recipe.clone(),
            space: comptime!(space.divide()),
            stage: comptime!(self.stage.descend()),
            _marker: PhantomData,
        }
    }

    pub(crate) fn evaluate(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        let mut absolute = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            absolute.push(self.origin.at(p) + pos.at(p));
        }
        self.recipe.evaluate(&absolute, space)
    }

    /// Evaluate with the static partial-tile mask. Dynamic axes are unmasked because a recipe
    /// has no source-local runtime extent for them.
    pub(crate) fn evaluate_masked(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        if comptime!(self.bounds_check) && !self.is_in_bounds(pos) {
            T::from_int(0)
        } else {
            self.evaluate(pos, space)
        }
    }

    /// Rebind an in-place staged payload to the source's current region. A recipe may hold runtime
    /// state, but the payload was derived from this very source by
    /// [`at_space`](Self::at_space), so both share one expansion-time recipe and only the origin
    /// can differ.
    pub(crate) fn rebind(&mut self, source: &Self) {
        self.origin.store_from(&source.origin);
    }

    /// Keep the recipe and current origin while changing the tile level that interprets its
    /// coordinates. An in-place staging payload is allocated for `Tile::divide()`, so its view
    /// must evaluate positions in that divided space rather than the source's parent space.
    pub(crate) fn at_space(&self, #[comptime] space: Space) -> Self {
        ProceduralData::<T> {
            origin: self.origin.stored(),
            bound: self.bound.clone(),
            bounds_check: comptime!(self.bounds_check),
            recipe: self.recipe.clone(),
            space,
            stage: comptime!(self.stage.descend()),
            _marker: PhantomData,
        }
    }

    pub(crate) fn evaluate_dyn(&self, pos: &CoordsDyn, #[comptime] space: Space) -> T {
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            coords.push(pos[p]);
        }
        self.evaluate(&coords, space)
    }

    fn is_in_bounds(&self, pos: &Coords<u32>) -> bool {
        let mut in_bounds = true;
        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            in_bounds = in_bounds && self.origin.at(p) + pos.at(p) < self.bound.at(p);
        }
        in_bounds
    }
}

impl<T: Numeric> Vectorized for ProceduralData<T> {}

impl<T: Numeric> VectorizedExpand for ProceduralDataExpand<T> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }
}

impl<T: Numeric, W: Size> ViewOperations<Vector<T, W>, CoordsDyn> for ProceduralData<T> {}

impl<T: Numeric, W: Size> ViewOperationsExpand<Vector<T, W>, CoordsDyn>
    for ProceduralDataExpand<T>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<T, W>> {
        let value = self
            .clone()
            .__expand_evaluate_dyn_method(scope, &pos, self.space.clone());
        Vector::<T, W>::__expand_cast_from(scope, value)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<T, W>> {
        let valid =
            <Self as ViewOperationsExpand<Vector<T, W>, CoordsDyn>>::__expand_is_in_bounds_method(
                self,
                scope,
                pos.clone(),
            );
        let value = self.__expand_read_method(scope, pos);
        let zero = Vector::<T, W>::__expand_cast_from(scope, 0.into());
        select::expand::<Vector<T, W>>(scope, valid, value, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
        mask_value: NativeExpand<Vector<T, W>>,
    ) -> NativeExpand<Vector<T, W>> {
        let valid =
            <Self as ViewOperationsExpand<Vector<T, W>, CoordsDyn>>::__expand_is_in_bounds_method(
                self,
                scope,
                pos.clone(),
            );
        let value = self.__expand_read_method(scope, pos);
        select::expand::<Vector<T, W>>(scope, valid, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<T, W>> {
        self.__expand_read_method(scope, pos)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <CoordsDyn as CubeType>::ExpandType,
        _end: <CoordsDyn as CubeType>::ExpandType,
    ) -> &SliceExpand<Vector<T, W>> {
        panic!("ProceduralData: procedural sources have no backing slice")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <CoordsDyn as CubeType>::ExpandType {
        CoordsDyn::__expand_new(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        let mut in_bounds: NativeExpand<bool> = true.into();
        for p in 0..comptime!(self.space.rank()) {
            let index = p.into_expand(scope);
            let origin = self.origin.__expand_at_method(scope, index);
            let bound = self.bound.__expand_at_method(scope, index);
            let pos = pos.clone();
            let coord = pos.__expand_index_method(scope, index);
            let absolute = origin.__expand_add_method(scope, *coord);
            let axis_in_bounds = absolute.__expand_lt_method(scope, &bound);
            in_bounds = in_bounds.__expand_and_method(scope, axis_in_bounds);
        }
        in_bounds
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<T, W>>,
        _pos: <CoordsDyn as CubeType>::ExpandType,
    ) {
        panic!("ProceduralData: procedural sources cannot issue TMA loads")
    }
}
