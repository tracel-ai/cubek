//! The unit routines: every unit owns queries of one head, or keys of one key head, and reads its
//! operands where they lie. They ask nothing of the hardware, so they run everywhere.

mod kernel;
mod key_gradient;
mod query_gradient;

pub use key_gradient::*;
pub use query_gradient::*;

use core::num::NonZeroUsize;
use cubecl::tile::space::UnitCube;
use cubek_std::Hardware;
use cubek_std::routine::CubeResources;

/// The cube a unit routine groups `planes` planes in. The units cooperate on nothing, so each
/// plane is as wide as the device dispatches one and no wider than the cube's unit budget leaves
/// for each: any width is correct.
fn unit_cube(hw: &Hardware, planes: NonZeroUsize) -> (UnitCube, CubeResources) {
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
