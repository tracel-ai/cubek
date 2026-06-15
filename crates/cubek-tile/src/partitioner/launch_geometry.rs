//! The launch geometry a [`Partitioner`] implies.

use cubecl::prelude::*;

use crate::Space;

use super::{ComputeScope, CubeAxis, Partitioner};

impl Partitioner {
    /// Cube dimension `d` gets the instance count of whichever axis is
    /// `Spatial { Cube(d), .. }`, at any level of the tree, else 1.
    pub fn cube_count(&self, space: &Space) -> CubeCount {
        CubeCount::Static(
            instances_for(space, ComputeScope::Cube(CubeAxis::X)),
            instances_for(space, ComputeScope::Cube(CubeAxis::Y)),
            instances_for(space, ComputeScope::Cube(CubeAxis::Z)),
        )
    }

    /// `plane_size × plane_count`. Plane length is the hardware's (`1` on CPU, the warp
    /// width on GPU).
    pub fn cube_dim<R: Runtime>(&self, client: &ComputeClient<R>, space: &Space) -> CubeDim {
        let plane_size = client.properties().hardware.plane_size_max;
        CubeDim::new_2d(plane_size, instances_for(space, ComputeScope::Plane))
    }
}

/// Product of instance counts over every axis riding `scope`, across the whole partitioner
/// tree (`1` when none). Each level contributes its spatial axes' instances; nested grids
/// multiply, and several axes sharing one dim ride it as a mixed-radix index — [`Walk`]
/// decodes the flat position back into per-axis coordinates. The tree is descended via
/// [`Space::divide`], so each level sees its own grid (parent edge ÷ this level's edge).
fn instances_for(space: &Space, scope: ComputeScope) -> u32 {
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
