use cubek::reduce::{
    launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const STRATEGY_UNIT_SERIAL: &str = "unit_serial";
pub const STRATEGY_UNIT_PARALLEL: &str = "unit_parallel";
pub const STRATEGY_PLANE_SERIAL: &str = "plane_serial";
pub const STRATEGY_PLANE_PARALLEL: &str = "plane_parallel";
pub const STRATEGY_CUBE_SERIAL: &str = "cube_serial";
pub const STRATEGY_CUBE_PARALLEL: &str = "cube_parallel";

pub fn strategies() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: STRATEGY_UNIT_SERIAL.to_string(),
            label: "Unit (serial)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_UNIT_PARALLEL.to_string(),
            label: "Unit (parallel)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_PLANE_SERIAL.to_string(),
            label: "Plane independent (serial)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_PLANE_PARALLEL.to_string(),
            label: "Plane independent (parallel)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_CUBE_SERIAL.to_string(),
            label: "Cube use_planes (serial)".to_string(),
        },
        ItemDescriptor {
            id: STRATEGY_CUBE_PARALLEL.to_string(),
            label: "Cube use_planes (parallel)".to_string(),
        },
    ]
}

pub(crate) fn strategy_for(id: &str) -> Option<ReduceStrategy> {
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
    Some(match id {
        STRATEGY_UNIT_SERIAL => ReduceStrategy {
            routine: unit(),
            vectorization: serial,
        },
        STRATEGY_UNIT_PARALLEL => ReduceStrategy {
            routine: unit(),
            vectorization: parallel,
        },
        STRATEGY_PLANE_SERIAL => ReduceStrategy {
            routine: plane(),
            vectorization: serial,
        },
        STRATEGY_PLANE_PARALLEL => ReduceStrategy {
            routine: plane(),
            vectorization: parallel,
        },
        STRATEGY_CUBE_SERIAL => ReduceStrategy {
            routine: cube(),
            vectorization: serial,
        },
        STRATEGY_CUBE_PARALLEL => ReduceStrategy {
            routine: cube(),
            vectorization: parallel,
        },
        _ => return None,
    })
}
