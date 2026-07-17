use cubecl::config::autotune::AutotuneLevel;
use cubek_test_utils::CatalogEntry;

use crate::launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy};
use crate::routines::{
    BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy,
};

pub fn strategies() -> Vec<CatalogEntry<ReduceStrategy>> {
    let unit = || RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy));
    let plane = || {
        RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: true,
        }))
    };
    let cube = || {
        RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: true,
        }))
    };
    let serial = VectorizationStrategy {
        parallel_output_vectorization: false,
    };
    let parallel = VectorizationStrategy {
        parallel_output_vectorization: true,
    };
    vec![
        CatalogEntry::new(
            "unit_serial",
            "Unit (serial)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: unit(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "unit_parallel",
            "Unit (parallel)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: unit(),
                vectorization: parallel,
            },
        ),
        CatalogEntry::new(
            "plane_serial",
            "Plane independent (serial)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: plane(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "plane_parallel",
            "Plane independent (parallel)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: plane(),
                vectorization: parallel,
            },
        ),
        CatalogEntry::new(
            "cube_serial",
            "Cube use_planes (serial)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: cube(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "cube_parallel",
            "Cube use_planes (parallel)",
            ReduceStrategy {
                autotune_level: AutotuneLevel::Full,
                routine: cube(),
                vectorization: parallel,
            },
        ),
    ]
}
