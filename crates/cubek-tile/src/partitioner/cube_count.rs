//! The launch geometry a [`Partitioner`] implies.

use cubecl::prelude::*;

use crate::Space;

use super::{ComputeScope, CubeAxis, Distribution, Partitioner};

/// The launch geometry a partitioner implies: cube dimension `d` gets the
/// instance count of whichever axis is `Spatial { Cube(d), .. }`, else 1.
pub fn cube_count_for(partitioner: &Partitioner, space: &Space) -> CubeCount {
    let instances_along = |dim: CubeAxis| -> u32 {
        let mut i = 0;
        while i < space.rank() {
            let axis = space.axis_at(i);
            if let Distribution::Spatial {
                scope: ComputeScope::Cube(cube_dim),
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
        instances_along(CubeAxis::X),
        instances_along(CubeAxis::Y),
        instances_along(CubeAxis::Z),
    )
}

/// The cube dimension a partitioner implies: a `Plane`-spread axis becomes the
/// cube's planes — one plane = one core on CPU (plane length 1) — each holding
/// `plane_size` units. Axes not spread across planes don't affect it; with no
/// `Plane` axis the cube is a single plane.
pub fn cube_dim_for<R: Runtime>(
    client: &ComputeClient<R>,
    partitioner: &Partitioner,
    space: &Space,
) -> CubeDim {
    let plane_size = client.properties().hardware.plane_size_max;

    let mut planes = 1u32;
    let mut uses_planes = false;
    let mut i = 0;
    while i < space.rank() {
        let axis = space.axis_at(i);
        if let Distribution::Spatial {
            scope: ComputeScope::Plane,
            coverage,
            ..
        } = partitioner.distribution(axis)
        {
            uses_planes = true;
            let grid = space.extent(axis) / partitioner.sub_tile_edge(axis);
            planes = coverage.instances(grid) as u32;
        }
        i += 1;
    }

    // Plane spreading puts one core per plane, reading its index off the unit
    // position — valid only when a plane is a single unit. Reject otherwise.
    assert!(
        !uses_planes || plane_size == 1,
        "Plane spreading assumes plane length 1 (one core per plane); \
         this backend reports plane_size = {plane_size}"
    );

    CubeDim::new_2d(plane_size, planes)
}
