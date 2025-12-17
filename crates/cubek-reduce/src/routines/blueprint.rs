use crate::{BoundChecks, LineMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceBlueprint {
    /// How vectorization was applied.
    pub line_mode: LineMode,
    /// The global blueprint for the kernel.
    pub global: GlobalReduceBlueprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalReduceBlueprint {
    Unit(UnitReduceBlueprint),
    Plane(PlaneReduceBlueprint),
    Cube(CubeBlueprint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// A single cube reduces a full vector.
pub struct CubeBlueprint {
    // Too much cubes are spawned, we should put some to idle.
    //
    // # Notes
    //
    // This only happens when we hit the hardware limit in spawning cubes on a single axis.
    pub cube_idle: bool,
    // There are too many units in a cube causing out-of-bound.
    //
    // # Notes
    //
    // There are never too many cubes spawned.
    pub bound_checks: BoundChecks,
    /// The number of accumulators in shared memory.
    pub num_shared_accumulators: u32,
    // Whether we use plane instructions to merge accumulators.
    pub use_planes: bool,
}

/// A single plane reduces a full vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceBlueprint {
    // Too much planes are spawned, we should put some to idle.
    pub plane_idle: bool,
    // There are too many units in a plane causing out-of-bound.
    pub bound_checks: BoundChecks,
    // Wheter all units in a plane work independently during the reduction.
    //
    // # Notes
    //
    // When this setting is turned on, there is an extra step at the end to merge accumulators
    // within the plane using plane instructions.
    pub independent: bool,
}

/// A single unit reduces a full vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitReduceBlueprint {
    // Too much units are spawned, we should put some to idle.
    pub unit_idle: bool,
}
