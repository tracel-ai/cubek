use crate::{BoundChecks, IdleMode, VectorizationMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceBlueprint {
    /// How vectorization was applied.
    pub vectorization_mode: VectorizationMode,
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
    /// When too many cubes are spawned, we should put some to idle.
    ///
    /// # Notes
    ///
    /// This only happens when we hit the hardware limit in spawning cubes on a single axis.
    pub cube_idle: IdleMode,
    /// There are too many units in a cube causing out-of-bound.
    ///
    /// # Notes
    ///
    /// There are never too many cubes spawned.
    pub bound_checks: BoundChecks,
    /// The number of accumulators in shared memory.
    pub num_shared_accumulators: usize,
    /// Whether we use plane instructions to merge accumulators.
    pub use_planes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneMergeStrategy {
    ///  All units in a plane work independently during the reduction
    ///  but merge their accumulators at the end
    Lazy,
    ///  There is a plane reduction at each iteration
    Eager,
}

/// A single plane reduces a full vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneReduceBlueprint {
    /// Too many planes are spawned, we should put some to idle.
    pub plane_idle: IdleMode,
    /// There are too many units in a plane causing out-of-bound.
    pub bound_checks: BoundChecks,
    /// Whether we recombine accumulators at each iteration or at the end only
    pub plane_merge_strategy: PlaneMergeStrategy,
    /// If true, we ceil the used cube_dim x to runtime plane_dim
    pub plane_dim_ceil: bool,
}

/// A single unit reduces a full vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitReduceBlueprint {
    // Too many units are spawned, we should put some to idle.
    pub unit_idle: IdleMode,
}
