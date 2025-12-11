use crate::routines::{cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy};
use cubecl::{features::Plane, prelude::*};

#[derive(Debug, Clone)]
pub enum ReduceStrategy {
    /// A unit is responsable to reduce a full vector.
    FullUnit(UnitStrategy),
    /// A plane is responsable to reduce a full vector.
    FullPlane(PlaneStrategy),
    /// A cube is responsable to reduce a full vector.
    FullCube(CubeStrategy),
}

pub(crate) fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
