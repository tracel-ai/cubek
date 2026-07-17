use crate::routines::{
    BlueprintStrategy, cube::CubeRoutine, plane::PlaneRoutine, unit::UnitRoutine,
};
use cubecl::{config::autotune::AutotuneLevel, features::Plane, prelude::*};

#[derive(Debug, Clone)]
pub struct ReduceStrategy {
    pub routine: RoutineStrategy,
    pub vectorization: VectorizationStrategy,
    /// The autotune level this launch's selection is cached under. Kernel
    /// selection is cached per *anchored* key everywhere below
    /// [`AutotuneLevel::Full`], so comptime fast paths derived from raw
    /// divisibility (unchecked bounds, no idle guards) are only stable — and
    /// only taken — at `Full`, where every raw shape is its own key. Callers
    /// benchmarking a specific variant outside autotune set it directly.
    pub autotune_level: AutotuneLevel,
}

#[derive(Debug, Clone)]
pub enum RoutineStrategy {
    /// A unit is responsible to reduce a full vector.
    Unit(BlueprintStrategy<UnitRoutine>),
    /// A plane is responsible to reduce a full vector.
    Plane(BlueprintStrategy<PlaneRoutine>),
    /// A cube is responsible to reduce a full vector.
    Cube(BlueprintStrategy<CubeRoutine>),
}

#[derive(Debug, Clone, Copy)]
pub struct VectorizationStrategy {
    /// When the vectorization is parallel, enable vectorization of the output so that each
    /// unit can perform N reductions, where N is the output `vector_size`.
    pub parallel_output_vectorization: bool,
}

pub(crate) fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
