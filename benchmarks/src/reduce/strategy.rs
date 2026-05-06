use cubek::reduce::{
    launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

use crate::registry::CatalogEntry;

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
                routine: unit(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "unit_parallel",
            "Unit (parallel)",
            ReduceStrategy {
                routine: unit(),
                vectorization: parallel,
            },
        ),
        CatalogEntry::new(
            "plane_serial",
            "Plane independent (serial)",
            ReduceStrategy {
                routine: plane(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "plane_parallel",
            "Plane independent (parallel)",
            ReduceStrategy {
                routine: plane(),
                vectorization: parallel,
            },
        ),
        CatalogEntry::new(
            "cube_serial",
            "Cube use_planes (serial)",
            ReduceStrategy {
                routine: cube(),
                vectorization: serial,
            },
        ),
        CatalogEntry::new(
            "cube_parallel",
            "Cube use_planes (parallel)",
            ReduceStrategy {
                routine: cube(),
                vectorization: parallel,
            },
        ),
    ]
}
