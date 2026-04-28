use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::CubeDimResource;

/// Identifies which compute primitive executes a tile matmul.
pub trait Scope: Clone + Copy + Send + Sync + 'static {
    /// Compute resource a single instance of this scope occupies.
    fn default_resource() -> CubeDimResource;
}

#[derive(Clone, Copy)]
pub struct Unit;
#[derive(Clone, Copy)]
pub struct Plane;
#[derive(Clone, Copy)]
pub struct Cube;

impl Scope for Unit {
    fn default_resource() -> CubeDimResource {
        CubeDimResource::Units(1)
    }
}
impl Scope for Plane {
    fn default_resource() -> CubeDimResource {
        CubeDimResource::Planes(1)
    }
}
impl Scope for Cube {
    fn default_resource() -> CubeDimResource {
        unimplemented!("Cube scope does not have a default cube-dim resource")
    }
}

/// Zero-sized comptime marker used to carry a [Scope] generic through [Tile].
#[derive(CubeType, Clone, Copy)]
pub struct ScopeMarker<Sc: Scope> {
    #[cube(comptime)]
    _phantom: PhantomData<Sc>,
}
