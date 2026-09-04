//! [`Tile`] constructors for the memory-free stores: an arbitrary recipe, and the
//! constants that are recipes of no inputs.

use cubecl::{ir::Scope, prelude::*, unexpanded};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// Create a scalar, memory-free tile over a logical space, evaluated where it is read at every
    /// level. Dynamic extents are supplied by another operand when an operation is walked; a
    /// procedural tile never witnesses them.
    fn procedural_virtual(#[comptime] space: Space, recipe: VirtualRecipe<T>) -> Self {
        Tile::<T> {
            tile_kind: TileKind::new_Procedural(ProceduralData::<T>::new_virtual(
                comptime!(space.clone()),
                recipe,
            )),
            space,
            descent: comptime!(Descent::default()),
        }
    }
}

impl<T: Numeric> Tile<T> {
    /// Create a coordinate-backed tile from an arbitrary procedural recipe. The concrete recipe
    /// is erased only while CubeCL expands this call.
    pub fn procedural<R: Recipe<T> + 'static>(_space: Space, _recipe: R) -> Self {
        unexpanded!()
    }

    pub fn __expand_procedural<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> TileExpand<T> {
        Self::__expand_procedural_virtual(
            scope,
            space,
            VirtualRecipe::<T>::__expand_new::<R>(scope, recipe),
        )
    }

    /// Create a procedural tile while preserving the recipe's factorization for contraction: the
    /// consumer sees one factor per contracted axis instead of one opaque field.
    pub fn procedural_separable<R: SeparableRecipe<T> + 'static>(_space: Space, _recipe: R) -> Self
    where
        R::ExpandType: SeparableRecipeAxisDependencies,
    {
        unexpanded!()
    }

    pub fn __expand_procedural_separable<R: SeparableRecipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> TileExpand<T>
    where
        R::ExpandType: SeparableRecipeOps<T>,
    {
        // A separable procedural tile is evaluated where it is read: staging a recipe into shared
        // memory would drop its factorization and normalization metadata without diagnostic.
        Self::__expand_procedural_virtual(
            scope,
            space,
            VirtualRecipe::<T>::__expand_new_separable::<R>(scope, recipe),
        )
    }

    /// Create a coordinate-backed tile yielding constant zero.
    pub fn zeros(_space: Space) -> Self {
        unexpanded!()
    }

    pub fn __expand_zeros(scope: &Scope, space: Space) -> TileExpand<T> {
        Self::__expand_procedural::<Zeros>(scope, space, ZerosExpand {})
    }

    /// Create a coordinate-backed tile yielding constant one.
    pub fn ones(_space: Space) -> Self {
        unexpanded!()
    }

    pub fn __expand_ones(scope: &Scope, space: Space) -> TileExpand<T> {
        Self::__expand_procedural::<Ones>(scope, space, OnesExpand {})
    }

    /// Create a coordinate-backed tile yielding a constant value.
    pub fn constant(_space: Space, _value: T) -> Self {
        unexpanded!()
    }

    pub fn __expand_constant(scope: &Scope, space: Space, value: NativeExpand<T>) -> TileExpand<T> {
        Self::__expand_procedural::<Constant<T>>(scope, space, ConstantExpand::<T> { value })
    }
}
