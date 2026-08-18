//! A memory-free tile source evaluated from logical coordinates.

use core::marker::PhantomData;

use cubecl::{
    prelude::{barrier::Barrier, *},
    std::tensor::{ViewOperations, ViewOperationsExpand, layout::CoordsDyn},
};

use crate::{Axis, Coords, Fold, FoldExpand, Region, Space, StagePlan};

/// Built-in recipes for a [`TileKind::Procedural`](crate::TileKind::Procedural) source.
///
/// Recipes are comptime values, not runtime callbacks. [`AxisProduct`](Self::AxisProduct)
/// deliberately preserves separability so a later operand reader can cache one factor per axis.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ProceduralRecipe {
    Zero,
    One,
    Uniform { denominator: usize },
    AxisIndex { axis: Axis },
    AxisProduct(Vec<ProceduralRecipe>),
}

impl ProceduralRecipe {
    pub fn zero() -> Self {
        Self::Zero
    }
    pub fn one() -> Self {
        Self::One
    }
    pub fn uniform(denominator: usize) -> Self {
        assert!(
            denominator > 0,
            "ProceduralRecipe::uniform: denominator must be non-zero"
        );
        Self::Uniform { denominator }
    }
    pub fn axis_index(axis: Axis) -> Self {
        Self::AxisIndex { axis }
    }
    pub fn axis_product(factors: Vec<ProceduralRecipe>) -> Self {
        Self::AxisProduct(factors)
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
    #[cube(comptime)]
    recipe: ProceduralRecipe,
    #[cube(comptime)]
    space: Space,
    /// Where this source lives at each level below. A recipe has no bytes to leave in place, so the
    /// default stages nothing; a level asking for a stage cooperatively materializes it
    /// ([`MemData::fill_procedural`](crate::MemData)).
    #[cube(comptime)]
    pub(crate) stage: StagePlan,
    #[cube(comptime)]
    _marker: PhantomData<T>,
}

#[cube]
impl<T: Numeric> ProceduralData<T> {
    pub(crate) fn new(
        #[comptime] space: Space,
        #[comptime] recipe: ProceduralRecipe,
        #[comptime] stage: StagePlan,
    ) -> Self {
        let mut origin = Coords::<u32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            origin.push(0u32.runtime());
            let axis = comptime!(space.axis_at(p));
            // Keep `bound` rank-aligned with `origin` even after `Space::divide` turns child
            // axes static. A recipe has no runtime bound for a dynamic axis, so its sentinel
            // deliberately leaves that axis unmasked.
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
            recipe: comptime!(self.recipe.clone()),
            space: comptime!(space.divide()),
            stage: comptime!(self.stage.descend()),
            _marker: PhantomData,
        }
    }

    pub(crate) fn evaluate(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        self.value_recipe(pos, space, comptime!(self.recipe.clone()))
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

    pub(crate) fn evaluate_dyn(&self, pos: &CoordsDyn, #[comptime] space: Space) -> T {
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            coords.push(pos[p]);
        }
        self.evaluate(&coords, space)
    }

    fn value_recipe(
        &self,
        pos: &Coords<u32>,
        #[comptime] space: Space,
        #[comptime] recipe: ProceduralRecipe,
    ) -> T {
        match comptime!(recipe) {
            ProceduralRecipe::Zero => T::from_int(0),
            ProceduralRecipe::One => T::from_int(1),
            ProceduralRecipe::Uniform { denominator } => {
                T::from_int(1) / T::from_int(denominator as i64)
            }
            ProceduralRecipe::AxisIndex { axis } => {
                let p = comptime!(space.position(axis));
                T::cast_from(self.origin.at(p) + pos.at(p))
            }
            ProceduralRecipe::AxisProduct(factors) => {
                let mut value = T::from_int(1);
                #[allow(clippy::needless_range_loop)]
                #[unroll]
                for i in 0..comptime!(factors.len()) {
                    value *= self.value_recipe(
                        pos,
                        comptime!(space.clone()),
                        comptime!(factors[i].clone()),
                    );
                }
                value
            }
        }
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
