use crate::routines::{
    BlueprintStrategy, cube::CubeRoutine, plane::PlaneRoutine, unit::UnitRoutine,
};
use cubecl::{features::Plane, prelude::*};

#[derive(Debug, Clone)]
pub struct ReduceStrategy {
    pub line_size: LineSizeStrategy,
    pub routine: RoutineStrategy,
}

#[derive(Debug, Clone)]
pub enum RoutineStrategy {
    /// A unit is responsable to reduce a full vector.
    Unit(BlueprintStrategy<UnitRoutine>),
    /// A plane is responsable to reduce a full vector.
    Plane(BlueprintStrategy<PlaneRoutine>),
    /// A cube is responsable to reduce a full vector.
    Cube(BlueprintStrategy<CubeRoutine>),
}

#[derive(Debug, Clone)]
pub struct LineSizeStrategy {
    /// When the vectorization is parallel, enable vectorization of the output so that each
    /// unit can perform N reductions, where N is the output `line_size`.
    pub parallel_output_vectorization: bool,
}

pub(crate) fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
