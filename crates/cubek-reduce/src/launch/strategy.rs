use crate::routines::{RoutineStrategy, cube::CubeRoutine, plane::PlaneRoutine, unit::UnitRoutine};
use cubecl::{features::Plane, prelude::*};

#[derive(Debug, Clone)]
pub enum ReduceStrategy {
    /// A unit is responsable to reduce a full vector.
    FullUnit(RoutineStrategy<UnitRoutine>),
    /// A plane is responsable to reduce a full vector.
    FullPlane(RoutineStrategy<PlaneRoutine>),
    /// A cube is responsable to reduce a full vector.
    FullCube(RoutineStrategy<CubeRoutine>),
}

pub(crate) fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
