//! The cube a unit routine groups its units in.

use core::num::NonZeroUsize;
use cubecl::tile::space::UnitCube;
use cubek_std::Hardware;
use cubek_std::routine::CubeResources;

/// The cube a unit routine groups `planes` planes in. The units cooperate on nothing, so each
/// plane is as wide as the device dispatches one and no wider than the cube's unit budget leaves
/// for each: any width is correct.
pub(super) fn unit_cube(hw: &Hardware, planes: NonZeroUsize) -> (UnitCube, CubeResources) {
    let planes = planes.get();
    let plane_width = (hw.properties.hardware.max_units_per_cube / planes as u32)
        .min(hw.dispatch_plane_size())
        .max(1);
    let cube = UnitCube {
        plane_width: plane_width as usize,
        planes,
    };
    let resources = CubeResources {
        plane_width,
        shared_memory: 0,
    };
    (cube, resources)
}
