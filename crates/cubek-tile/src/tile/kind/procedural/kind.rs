use core::marker::PhantomData;

use cubecl::frontend::IntoExpand;
use cubecl::ir::Scope;
use cubecl::unexpanded;
use cubecl::{
    prelude::{barrier::Barrier, *},
    std::tensor::{ViewOperations, ViewOperationsExpand, layout::CoordsDyn},
};

use crate::*;

use super::{ErasedRecipe, FactorReads, RecipeCoords, Separable, SeparableCall};

/// Runtime state of a procedural source. `origin` tracks regions selected by `Tile::at`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Procedural<T: Numeric> {
    origin: Coords<u32>,
    /// The source's static logical extent. Dynamic axes hold `u32::MAX`, deliberately leaving
    /// them unmasked; unlike `space`, this stays in the parent's coordinate system as
    /// [`Tile::at`](crate::Tile::at) descends into nominally sized trailing partial tiles.
    bound: Coords<u32>,
    /// Whether any level can select a partial tile. It stays with the source while its `space`
    /// descends, because the leaf space alone no longer records an ancestor's overhang.
    #[cube(comptime)]
    pub(crate) bounds_check: bool,
    /// The statically overhanging axes. Keeping this comptime avoids emitting a per-tap bound
    /// comparison when only an unrelated axis has a trailing partial tile.
    #[cube(comptime)]
    bounded_axes: Vec<Axis>,
    /// Requested factor normalization and the space whose complete factor runs it describes. Only
    /// a separable contraction consumes it, since only that leaf knows each factor's tap run; the
    /// original space lets it reject an ancestor split that would normalize each chunk on its own.
    #[cube(comptime)]
    pub(crate) normalization: Option<Normalization>,
    recipe: ErasedRecipe<T>,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    _marker: PhantomData<T>,
}

#[cube]
impl<T: Numeric> Procedural<T> {
    #[allow(dead_code)] // Reached through its expand, from [`Tile::procedural`].
    pub(crate) fn erased(#[comptime] space: Space, recipe: ErasedRecipe<T>) -> Self {
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
        // Nothing has cut this tile yet: an axis overhangs once a level's edge fails to divide
        // it, which `at` records on the way down.
        let bounded_axes = comptime!(Vec::new());
        Procedural::<T> {
            origin,
            bound,
            bounds_check: comptime!(false),
            bounded_axes,
            normalization: None,
            recipe,
            space,
            _marker: PhantomData,
        }
    }

    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> Self {
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            match comptime!(step.level.tile(axis)) {
                Some(tile) => {
                    let tile = comptime!(tile as u32);
                    origin.push(self.origin.at(p) + step.coord(axis).cast::<u32>() * tile);
                }
                None => origin.push(self.origin.at(p)),
            }
        }
        // An axis this level cuts unevenly leaves a partial tile below; from here down every
        // read along it is checked.
        let bounded_axes = comptime!({
            let mut axes = self.bounded_axes.clone();
            for axis in space.axes() {
                if !space.is_dynamic(axis)
                    && step.level.overhangs(&space, axis)
                    && !axes.contains(&axis)
                {
                    axes.push(axis);
                }
            }
            axes
        });
        Procedural::<T> {
            origin,
            bound: self.bound.clone(),
            bounds_check: comptime!(!bounded_axes.is_empty()),
            bounded_axes,
            normalization: comptime!(self.normalization.clone()),
            recipe: self.recipe.clone(),
            space: comptime!(step.level.child(&space)),
            _marker: PhantomData,
        }
    }

    pub(crate) fn evaluate(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        let absolute = RecipeCoords::new(&self.origin, pos, space);
        self.recipe.evaluate(&absolute)
    }

    pub(crate) fn factorization(&self) -> comptime_type!(Option<usize>) {
        self.recipe.factorization()
    }

    #[allow(dead_code)] // Reached through its expand, from [`Tile::factor_dependencies`].
    pub(crate) fn factor_reads(
        &self,
        #[comptime] factor: usize,
        #[comptime] axis: Axis,
    ) -> comptime_type!(bool) {
        self.recipe.factor_reads(factor, axis)
    }

    pub(crate) fn read_factor_at(
        &self,
        pos: &CoordsDyn,
        #[comptime] factor: usize,
        #[comptime] space: Space,
    ) -> T {
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            coords.push(pos[p]);
        }
        let absolute = RecipeCoords::new(&self.origin, &coords, space);
        self.recipe.factor(&absolute, factor)
    }

    /// Whether `pos` remains inside the original procedural box on one logical axis. Factor-local
    /// normalization asks only about the axis its tap moves, so another factor's placeholder
    /// coordinate cannot mask this one.
    pub(crate) fn axis_in_bounds(&self, pos: &CoordsDyn, #[comptime] axis: Axis) -> bool {
        if comptime!(self.bounded_axes.contains(&axis) && self.space.contains(axis)) {
            let p = comptime!(self.space.position(axis));
            self.origin.at(p) + pos[p] < self.bound.at(p)
        } else {
            true.runtime()
        }
    }

    /// Evaluate with the static partial-tile mask. Dynamic axes are unmasked because a recipe
    /// has no source-local runtime extent for them.
    pub(crate) fn read_masked(&self, pos: &Coords<u32>, #[comptime] space: Space) -> T {
        if comptime!(self.bounds_check) && !self.is_in_bounds(pos) {
            T::from_int(0)
        } else {
            self.evaluate(pos, space)
        }
    }

    #[allow(dead_code)] // Reached through its expand, from the `ViewOperationsExpand` impl below.
    pub(crate) fn read_at(&self, pos: &CoordsDyn, #[comptime] space: Space) -> T {
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

impl<T: Numeric> Vectorized for Procedural<T> {}

impl<T: Numeric> ProceduralExpand<T> {
    pub(crate) fn factor_count(&self, scope: &Scope) -> Option<usize> {
        self.recipe.__expand_factorization_method(scope)
    }
}

impl<T: Numeric> VectorizedExpand for ProceduralExpand<T> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }
}

impl<T: Numeric, W: Size> ViewOperations<Vector<T, W>, CoordsDyn> for Procedural<T> {}

impl<T: Numeric, W: Size> ViewOperationsExpand<Vector<T, W>, CoordsDyn> for ProceduralExpand<T> {
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> NativeExpand<Vector<T, W>> {
        assert_eq!(
            W::__expand_value(scope),
            1,
            "Procedural: a procedural read is scalar; a vectorized read would broadcast \
             the first unit's value instead of ramping the innermost coordinate"
        );
        let value = self
            .clone()
            .__expand_read_at_method(scope, &pos, self.space.clone());
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
        panic!("Procedural: procedural sources have no backing slice")
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
        panic!("Procedural: procedural sources cannot issue TMA loads")
    }
}

#[cube]
impl<T: Numeric> Procedural<T> {
    /// This source as a tile, placed alone: a recipe is evaluated where it is read, so no level
    /// above it cuts it until a walk states one.
    pub fn tile(self) -> Tile<T> {
        let space = comptime!(self.space.clone());
        Tile::new(
            TileKind::new_Procedural(self),
            comptime!(Placement::alone(space)),
        )
    }
}

impl<T: Numeric> Procedural<T> {
    /// A memory-free source over `space`, evaluated from `recipe` at the logical coordinates it
    /// is read at. The concrete recipe is erased while the kernel expands and nowhere else.
    ///
    /// Dynamic extents are supplied by another operand when an operation is walked; a procedural
    /// source never witnesses them.
    pub fn new<R: Recipe<T> + 'static>(_space: Space, _recipe: R) -> Self {
        unexpanded!()
    }

    pub fn __expand_new<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> ProceduralExpand<T> {
        Self::__expand_erased(
            scope,
            space,
            ErasedRecipe::<T>::__expand_new::<R>(scope, recipe),
        )
    }

    /// [`new`](Self::new) keeping the recipe's factorization: a consumer sees one factor per
    /// contracted axis instead of one opaque field, and evaluates each factor once per tap run.
    pub fn separable<R: Separable<T> + 'static>(_space: Space, _recipe: R) -> Self
    where
        R::ExpandType: FactorReads,
    {
        unexpanded!()
    }

    pub fn __expand_separable<R: Separable<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> ProceduralExpand<T>
    where
        R::ExpandType: SeparableCall<T>,
    {
        // A separable source is evaluated where it is read: staging a recipe into shared memory
        // would drop its factorization and its normalization without diagnostic.
        Self::__expand_erased(
            scope,
            space,
            ErasedRecipe::<T>::__expand_new_separable::<R>(scope, recipe),
        )
    }
}

impl<T: Float> Procedural<T> {
    /// Normalize this source's factor runs where the gather contraction evaluates them: each
    /// factor's taps are summed and divided out there, since only that leaf knows a tap run.
    ///
    /// Refused for a recipe that states no factorization: a post-pass over the values would hide
    /// a second walk. Summing only the taps in bounds ([`TapSupport::InBounds`]) also needs the
    /// operand read at its source window, so staging it in shared memory is refused at launch.
    pub fn normalized(self, _taps: TapSupport, _guard: DivGuard) -> Self {
        unexpanded!()
    }
}

impl<T: Float> ProceduralExpand<T> {
    pub fn __expand_normalized_method(
        mut self,
        scope: &Scope,
        taps: TapSupport,
        guard: DivGuard,
    ) -> ProceduralExpand<T> {
        assert!(
            self.factor_count(scope).is_some(),
            "Procedural::normalized: the recipe states no separable factorization"
        );
        self.normalization = Some(Normalization::new(taps, guard, self.space.clone()));
        self
    }
}

#[cfg(test)]
mod tests {
    use cubecl::ir::{ExpandValue, Scope};

    use crate::*;

    fn test_scope() -> Scope {
        Scope::root(cubecl::ir::settings::KernelSettings::new(
            cubecl::ir::settings::Dim3::new_single(),
            cubecl::ir::settings::ExecutionMode::Checked,
            cubecl::ir::AddressType::U32,
        ))
    }

    #[test]
    #[should_panic(expected = "the recipe states no separable factorization")]
    fn normalized_rejects_an_opaque_recipe() {
        let scope = test_scope();
        let source = Procedural::<f32>::__expand_new::<Constant<f32>>(
            &scope,
            Space::new(&[(Axis(0), 4)]),
            ConstantExpand::<f32> {
                value: ExpandValue::constant(
                    0u64.into(),
                    cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
                )
                .into(),
            },
        );
        source.__expand_normalized_method(&scope, TapSupport::Whole, DivGuard::default());
    }
}
