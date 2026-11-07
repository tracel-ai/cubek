//! Which random routine fills an output, and at what settings.

use crate::kernels::{PlaneRandomRoutine, RandomKernel, RandomStrategy, UnitRandomRoutine};
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// The kernel `strategy` runs on `hw`.
///
/// On a GPU the plane is the vector unit: neighbouring lanes already store one contiguous stretch
/// per draw, and the parallelism is the grid. On a CPU a unit is a core, and two cores must never
/// write the same cache line, so each takes a run of its own.
pub fn select(strategy: RandomStrategy, hw: &Hardware) -> RandomKernel {
    match (strategy, hw.properties.hardware.num_cpu_cores) {
        (RandomStrategy::Specific(kernel), _) => kernel,
        (RandomStrategy::Selected, Some(_)) => {
            RandomKernel::Unit(UnitRandomRoutine.settings(hw)[0])
        }
        (RandomStrategy::Selected, None) => RandomKernel::Plane(PlaneRandomRoutine.settings(hw)[0]),
    }
}
