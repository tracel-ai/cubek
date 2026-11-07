use cubek_test_utils::CatalogEntry;

use crate::kernels::{PlaneRandomRoutine, RandomKernel, RandomStrategy, UnitRandomRoutine};
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// The selected kernel, then each routine at its default on the test device.
pub fn strategies() -> Vec<CatalogEntry<RandomStrategy>> {
    let hw = &Hardware::from_client(&cubecl::test_device().client());
    vec![
        CatalogEntry::new("auto", "Auto", RandomStrategy::Selected),
        CatalogEntry::new(
            "plane",
            "Plane",
            RandomStrategy::Specific(RandomKernel::Plane(PlaneRandomRoutine.settings(hw)[0])),
        ),
        CatalogEntry::new(
            "unit",
            "Unit",
            RandomStrategy::Specific(RandomKernel::Unit(UnitRandomRoutine.settings(hw)[0])),
        ),
    ]
}
