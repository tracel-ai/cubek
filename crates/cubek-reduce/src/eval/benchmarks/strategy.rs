use cubecl::config::autotune::AutotuneLevel;
use cubek_test_utils::CatalogEntry;

use crate::eval::cpu_reference::ReduceEntry;
use crate::kernels::{ReduceKernel, ReduceKernelHeuristic, ReduceOpStrategy};
use crate::launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy};
use crate::routines::{
    BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy,
};

/// Today's kernels at each routine and vectorization, then the tile routines at every strategy:
/// the autotune, the rule, and each kernel pinned.
pub fn strategies() -> Vec<CatalogEntry<ReduceEntry>> {
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
    let kernels = vec![
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
    ];
    let tiles = core::iter::once(ReduceOpStrategy::Autotune)
        .chain([ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile)])
        .chain(
            ReduceKernel::all()
                .into_iter()
                .map(ReduceOpStrategy::Specific),
        )
        .map(|strategy| {
            let id = format!("tiles_{}", strategy.name().replace('-', "_"));
            let label = format!("Tiles {}", strategy.name());
            CatalogEntry::new(id, label, ReduceEntry::Tiles(strategy))
        });
    let shared_sums = [4, 16, 64].map(|cubes| {
        CatalogEntry::new(
            format!("kernels_shared_sum_{cubes}"),
            format!("shared_sum, {cubes} cubes (whole-tensor sums only)"),
            ReduceEntry::SharedSum(cubes),
        )
    });
    kernels
        .into_iter()
        .map(|entry| CatalogEntry::new(entry.id, entry.label, ReduceEntry::Kernels(entry.value)))
        .chain(tiles)
        .chain(shared_sums)
        .collect()
}
