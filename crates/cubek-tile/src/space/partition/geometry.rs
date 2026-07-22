//! The launch geometry a [`Space`]'s partitioner tree implies.

use cubecl::prelude::*;

use crate::Space;

use super::{ComputeScope, CubeAxis};

impl Space {
    /// Cube dimension `d` gets the instance count of whichever axis is
    /// `Spatial { Cube(d), .. }`, at any level of the tree, else 1.
    pub fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            instances_count(self, ComputeScope::Cube(CubeAxis::X)),
            instances_count(self, ComputeScope::Cube(CubeAxis::Y)),
            instances_count(self, ComputeScope::Cube(CubeAxis::Z)),
        )
    }

    /// `plane_size × plane_count`. Plane length is the hardware's (`1` on CPU, the warp
    /// width on GPU). `Unit` axes ride those `plane_size` lanes, so their instance product
    /// must be exactly `plane_size` (or `1`, no Unit split); a badly sized split would idle
    /// or race lanes. A deferred `PlaneLanes` count panics here: launch through
    /// [`launcher`](Space::launcher), which stamps it.
    pub fn cube_dim<R: Runtime>(&self, client: &ComputeClient<R>) -> CubeDim {
        let plane_size = client.properties().hardware.plane_size_max;
        let lanes = instances_count(self, ComputeScope::Unit);
        assert!(
            lanes == 1 || lanes == plane_size,
            "cube_dim: Unit axes must partition exactly plane_size ({plane_size}) lanes, got {lanes}"
        );
        CubeDim::new_2d(plane_size, instances_count(self, ComputeScope::Plane))
    }
}

/// Product of instance counts over every axis riding `scope`, across the whole partitioner tree
fn instances_count(space: &Space, scope: ComputeScope) -> u32 {
    let mut total = 1u32;
    let mut level = space.clone();
    while !level.is_final() {
        for axis in level.axes() {
            let dist = level.partitioner().distribution(axis);
            if dist.scope() == Some(scope) {
                // `count` is `ceil`, so an indivisible axis adds the instance for its
                // partial tile.
                total *= dist.coverage().instances(level.count(axis)) as u32;
            }
        }
        level = level.divide();
    }
    total
}
