//! The launch geometry a [`Partitioner`] implies.

use cubecl::prelude::*;

use crate::Space;

use super::{ComputePrimitive, CubeDimension, Distribution, Partitioner};

/// The launch geometry a partitioner implies: cube dimension `d` gets the
/// instance count of whichever axis is `Spatial { Cube(d), .. }`, else 1.
pub fn cube_count_for(partitioner: &Partitioner, space: &Space) -> CubeCount {
    let instances_along = |dim: CubeDimension| -> u32 {
        let mut i = 0;
        while i < space.rank() {
            let axis = space.axis_at(i);
            if let Distribution::Spatial {
                unit: ComputePrimitive::Cube(cube_dim),
                coverage,
                ..
            } = partitioner.distribution(axis)
                && cube_dim == dim
            {
                let grid = space.extent(axis) / partitioner.sub_tile_edge(axis);
                return coverage.instances(grid) as u32;
            }
            i += 1;
        }
        1
    };
    CubeCount::Static(
        instances_along(CubeDimension::X),
        instances_along(CubeDimension::Y),
        instances_along(CubeDimension::Z),
    )
}
