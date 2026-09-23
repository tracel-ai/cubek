//! What a memory-free source is made of: a recipe evaluated at logical coordinates, and the
//! factorization one states when its factors read orthogonal axes.

use cubecl::ir::Scope;
use cubecl::prelude::*;

use crate::{Axis, Coords, Space};

/// Coordinate dependence of a recipe, queried while a kernel is expanded.
///
/// This deliberately lives on expand types: axes are compile-time state of recipe values, not a
/// runtime GPU value, and composing the answer in ordinary Rust avoids turning a collected list
/// of axes into mutable kernel state.
pub trait RecipeAxisDependencies {
    fn reads_axis(&self, scope: &Scope, axis: Axis) -> bool;
}

/// Per-factor coordinate dependencies of a separable recipe, queried during expansion.
pub trait SeparableRecipeAxisDependencies {
    fn factor_reads_axis(&self, scope: &Scope, factor: usize, axis: Axis) -> bool;
}

/// The absolute logical coordinates a [`Recipe`] is evaluated at: the source's `origin` plus the
/// position it is read at, within its [`Space`]. Rebased one axis at a time on demand, so a
/// recipe emits an add only for the axes it reads, and one that ignores its coordinates emits none.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RecipeCoords {
    origin: Coords<u32>,
    offset: Coords<u32>,
    #[cube(comptime)]
    pub space: Space,
}

#[cube]
impl RecipeCoords {
    pub(crate) fn new(
        origin: &Coords<u32>,
        offset: &Coords<u32>,
        #[comptime] space: Space,
    ) -> Self {
        RecipeCoords {
            origin: origin.clone(),
            offset: offset.clone(),
            space,
        }
    }

    /// The absolute coordinate along the axis at comptime position `p`.
    pub fn at(&self, #[comptime] p: usize) -> u32 {
        self.origin.at(p) + self.offset.at(p)
    }

    /// The absolute coordinate along the specified `axis`.
    pub fn along(&self, #[comptime] axis: Axis) -> u32 {
        self.at(comptime!(self.space.position(axis)))
    }
}

/// An N-dimensional scalar field evaluated at absolute logical coordinates.
///
/// Recipes implement [`Recipe<T>`] for any numeric element type `T: Numeric` (integers and floats),
/// though continuous interpolation and filtering recipes (such as [`Linear`](super::Linear),
/// [`Cubic`](super::Cubic), [`Lanczos`](super::Lanczos)) are defined over [`Float`] elements.
#[cube(expand_base_traits = "ExpandTypeClone")]
pub trait Recipe<T: Numeric> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T;
}

/// A recipe that factorizes into one factor per contracted axis, `R(coords) = ∏ᵢ Rᵢ(coords)`,
/// factor `i` varying only along the `i`-th contracted axis, so the gather microkernel evaluates
/// each factor once per 1-D tap walk instead of the whole product at every point of their product.
///
/// The factor count is the recipe's, not the consumer's: a 1-D, 2-D or N-D filter is the same
/// contract with a different `factors`.
#[cube]
pub trait SeparableRecipe<T: Numeric>: Recipe<T> {
    fn factors(&self) -> comptime_type!(usize);
    /// Evaluate the factor at comptime position `factor`, which indexes the contracted axes in
    /// the order the contraction walks them.
    fn evaluate_factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T;
}
