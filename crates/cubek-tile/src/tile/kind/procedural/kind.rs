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
    /// The source's static extent in the parent's coordinates; dynamic axes hold `u32::MAX`.
    bound: Coords<u32>,
    /// Whether any level can select a partial tile.
    #[cube(comptime)]
    pub(crate) bounds_check: bool,
    /// The statically overhanging axes.
    #[cube(comptime)]
    bounded_axes: Vec<Axis>,
    /// Requested factor normalization, consumed by a separable contraction.
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

    /// This recipe's value at cell `column` of row `row` of a tile over `space`, windowed to the
    /// same region: the row unravelled over every axis of `space` but the innermost, the column
    /// along the innermost, and an axis `space` does not span at zero.
    pub(crate) fn at_cell(&self, row: u32, column: u32, #[comptime] space: Space) -> T {
        // Where each of this recipe's axes reads its coordinate from: the column, a digit of the
        // row, or nowhere.
        let (row_extents, sources) = comptime!({
            let rank = space.rank();
            let extents: Vec<usize> = (0..rank - 1).map(|p| space.extent_at(p)).collect();
            let sources: Vec<Option<usize>> = self
                .space
                .axes()
                .map(|axis| match space.contains(axis) {
                    true => Some(space.position(axis)),
                    false => None,
                })
                .collect();
            (extents, sources)
        });
        let columns_at = comptime!(space.rank() - 1);
        let row_digits = Coords::<u32>::constant(row_extents).unravel(row);
        let mut coords = Coords::<u32>::new();
        // `#[unroll]` needs a range loop.
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for p in 0..sources.len() {
            match sources[p] {
                Some(at) if at == columns_at => coords.push(column),
                Some(at) => coords.push(row_digits.at(at)),
                None => coords.push(0u32.runtime()),
            }
        }
        self.evaluate(&coords, comptime!(self.space.clone()))
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

    /// Whether `pos` remains inside the original procedural box along `axis`.
    pub(crate) fn axis_in_bounds(&self, pos: &CoordsDyn, #[comptime] axis: Axis) -> bool {
        if comptime!(self.bounded_axes.contains(&axis) && self.space.contains(axis)) {
            let p = comptime!(self.space.position(axis));
            self.origin.at(p) + pos[p] < self.bound.at(p)
        } else {
            true.runtime()
        }
    }

    /// Evaluate with the static partial-tile mask; dynamic axes are unmasked.
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
    /// This source as a tile, placed alone.
    pub fn tile(self) -> Tile<T> {
        let space = comptime!(self.space.clone());
        Tile::new(
            TileKind::new_Procedural(self),
            comptime!(Placement::alone(space)),
        )
    }

    /// This source as a tile at the root of `partitioning`, windowed by its regions like an
    /// operand bound to it: its space must be the partitioning's, or some of its axes.
    pub fn tile_in(self, partitioning: &Partitioning) -> Tile<T> {
        let space = comptime!(self.space.clone());
        let levels = comptime!(partitioning.levels().to_vec());
        Tile::new(
            TileKind::new_Procedural(self),
            comptime!(Placement::root(space, levels)),
        )
    }
}

impl<T: Numeric> Procedural<T> {
    /// A memory-free source over `space`, evaluated from `recipe` where it is read.
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

    /// [`new`](Self::new) keeping the recipe's factorization, one factor per contracted axis.
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
        Self::__expand_erased(
            scope,
            space,
            ErasedRecipe::<T>::__expand_new_separable::<R>(scope, recipe),
        )
    }
}

impl<T: Float> Procedural<T> {
    /// Normalize each factor's taps where the gather contraction evaluates them.
    /// The recipe must state a factorization.
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

    use crate::tile::kind::procedural::constant::{Constant, ConstantExpand};
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
