//! The launch geometry a [`Partitioner`] implies.

use cubecl::prelude::*;

use crate::{Axis, Space};

use super::{ComputeScope, CubeAxis, Partitioner};

impl Partitioner {
    /// Cube dimension `d` gets the instance count of whichever axis is
    /// `Spatial { Cube(d), .. }`, else 1.
    pub fn cube_count(&self, space: &Space) -> CubeCount {
        CubeCount::Static(
            self.cube_instances(space, CubeAxis::X),
            self.cube_instances(space, CubeAxis::Y),
            self.cube_instances(space, CubeAxis::Z),
        )
    }

    /// `plane_size × plane_count`. Plane length is the hardware's (`1` on CPU, the warp
    /// width on GPU).
    pub fn cube_dim<R: Runtime>(&self, client: &ComputeClient<R>, space: &Space) -> CubeDim {
        let plane_size = client.properties().hardware.plane_size_max;
        CubeDim::new_2d(plane_size, self.plane_count(space))
    }

    /// Product of instance counts over every `Plane`-spread axis (`1` when none). One flat
    /// count that [`Walk`](crate::Walk) decodes back into per-axis coordinates.
    fn plane_count(&self, space: &Space) -> u32 {
        space
            .axes()
            .filter(|&axis| self.distribution(axis).scope() == Some(ComputeScope::Plane))
            .map(|axis| self.instances_along(space, axis))
            .product()
    }

    /// Instance count of `Cube(dim)`: the product over every axis riding it (several
    /// may share one dim as a mixed-radix index), or 1 if none do.
    fn cube_instances(&self, space: &Space, dim: CubeAxis) -> u32 {
        space
            .axes()
            .filter(|&axis| self.distribution(axis).scope() == Some(ComputeScope::Cube(dim)))
            .map(|axis| self.instances_along(space, axis))
            .product()
    }

    fn instances_along(&self, space: &Space, axis: Axis) -> u32 {
        // `ceil` so an indivisible axis launches the extra instance for its partial tile.
        let grid = space.extent(axis).div_ceil(self.edge(axis));
        self.distribution(axis).coverage().instances(grid) as u32
    }
}
