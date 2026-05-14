use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::CubeDimResource;

/// Identifies which compute primitive executes a tile matmul.
pub trait TileScope: Clone + Copy + Send + Sync + 'static {
    /// Compute resource a single instance of this scope occupies.
    fn default_resource() -> CubeDimResource;

    /// Comptime tag used at dispatch sites that need to assert a particular scope
    /// (e.g. variants that only make sense on a plane).
    const KIND: ScopeKind;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ScopeKind {
    Unit,
    Plane,
    Cube,
}

#[derive(Clone, Copy)]
pub struct Unit;
#[derive(Clone, Copy)]
pub struct Plane;
#[derive(Clone, Copy)]
pub struct Cube;

impl TileScope for Unit {
    fn default_resource() -> CubeDimResource {
        CubeDimResource::Units(1)
    }
    const KIND: ScopeKind = ScopeKind::Unit;
}
impl TileScope for Plane {
    fn default_resource() -> CubeDimResource {
        CubeDimResource::Planes(1)
    }
    const KIND: ScopeKind = ScopeKind::Plane;
}
impl TileScope for Cube {
    fn default_resource() -> CubeDimResource {
        unimplemented!("Cube scope does not have a default cube-dim resource")
    }
    const KIND: ScopeKind = ScopeKind::Cube;
}

/// Zero-sized comptime marker used to carry a [TileScope] generic through
/// [`Tile`](crate::tile::Tile) at the field level.
#[derive(CubeType, Clone, Copy)]
pub struct ScopeMarker<Sc: TileScope> {
    #[cube(comptime)]
    pub _phantom: PhantomData<Sc>,
}

/// Comptime assertion that a tile-scope generic resolves to `Plane`.
pub fn assert_plane_scope(kind: ScopeKind) {
    match kind {
        ScopeKind::Plane => {}
        _ => panic!("This Tile variant is only valid in Plane scope"),
    }
}

/// Comptime assertion that a tile-scope generic resolves to `Unit`.
pub fn assert_unit_scope(kind: ScopeKind) {
    match kind {
        ScopeKind::Unit => {}
        _ => panic!("This Tile variant is only valid in Unit scope"),
    }
}
