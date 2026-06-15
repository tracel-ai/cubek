use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::{
    BoundChecks, IdleMode, ReduceError, VectorizationMode,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, CubeBlueprint, Routine},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, features::Plane};
use cubek_std::cube_count::cube_count_spread_with_total;

#[derive(Debug, Clone)]
pub struct CubeRoutine;

#[derive(Debug, Clone)]
pub struct CubeStrategy {
    /// If we use plane to aggregate accumulators.
    pub use_planes: bool,
}

impl Routine for CubeRoutine {
    type Strategy = CubeStrategy;
    type Blueprint = CubeBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, cube_dim, num_cubes) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                // One accumulator per plane.
                if blueprint.use_planes {
                    if !support_plane(client) {
                        return Err(ReduceError::PlanesUnavailable);
                    }

                    if blueprint.num_shared_accumulators != cube_dim.x as usize {
                        return Err(ReduceError::Validation {
                            details: "Num accumulators should match cube_dim.x",
                        });
                    }
                    if cube_dim.x != client.properties().hardware.plane_size_max {
                        return Err(ReduceError::Validation {
                            details: "`cube_dim.x` must match `plane_size_max`",
                        });
                    }
                // One accumulator per unit.
                } else if blueprint.num_shared_accumulators != cube_dim.num_elems() as usize {
                    return Err(ReduceError::Validation {
                        details: "Num accumulators should match cube_dim.num_elems()",
                    });
                }

                // Reject a forced blueprint whose accumulators exceed the device's
                // shared-memory limit.
                let bytes_per_accumulator = bytes_per_accumulator(&problem, &settings);
                let requested = bytes_per_accumulator * blueprint.num_shared_accumulators;
                let available = client.properties().hardware.max_shared_memory_size;
                if requested > available {
                    return Err(ReduceError::SharedMemoryOverflow {
                        requested,
                        available,
                    });
                }

                let working_cubes = working_cubes(&settings, &problem);
                let (cube_count, launched_cubes) =
                    cube_count_spread_with_total(client, working_cubes);

                if working_cubes != launched_cubes && !blueprint.cube_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many cubes launched for the problem causing OOD, but `cube_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
                    global: GlobalReduceBlueprint::Cube(blueprint),
                };

                (blueprint, cube_dim, cube_count)
            }
            BlueprintStrategy::Inferred(strategy) => {
                let (blueprint, cube_dim, cube_count) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, cube_dim, cube_count)
            }
        };

        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count: num_cubes,
            address_type,
            vector: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceVectorSettings,
    strategy: CubeStrategy,
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let hardware_properties = &client.properties().hardware;
    let plane_size = hardware_properties.plane_size_max;

    let use_planes = strategy.use_planes
        && hardware_properties.plane_size_max == hardware_properties.plane_size_min;

    let working_cubes = working_cubes(settings, &problem);
    let working_units = working_cubes * problem.reduce_len.div_ceil(settings.vector_size_input);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, hardware_properties);

    // Occupancy sizes the cube above; shrink the plane count if the resulting
    // shared-memory footprint (which scales with `k` for `ArgTopK` / `TopK`) would
    // overrun the device limit, keeping `cube_dim.x = plane_size`.
    let plane_count = clamp_plane_count(
        bytes_per_accumulator(&problem, settings),
        client.properties().hardware.max_shared_memory_size,
        plane_size,
        plane_count,
        use_planes,
    )?;

    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let cube_size = cube_dim.num_elems();

    let work_size = match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_len / settings.vector_size_input,
        VectorizationMode::Perpendicular => problem.reduce_len,
    };
    let bound_checks = match work_size.is_multiple_of(cube_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let num_shared_accumulators = match use_planes {
        true => plane_count as usize,
        false => cube_size as usize,
    };

    let (cube_count, launched_cubes) = cube_count_spread_with_total(client, working_cubes);

    let cube_idle = match working_cubes != launched_cubes {
        true => match strategy.use_planes
            && !client
                .properties()
                .features
                .plane
                .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Mask,
            false => IdleMode::Terminate,
        },
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Cube(CubeBlueprint {
            cube_idle,
            bound_checks,
            num_shared_accumulators,
            use_planes,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_cubes(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.reduce_count / settings.vector_size_input,
    }
}

/// Shared bytes per accumulator slot for this problem's instruction.
fn bytes_per_accumulator(problem: &ReduceProblem, settings: &ReduceVectorSettings) -> usize {
    problem.instruction.shared_memory_bytes_per_accumulator(
        problem.dtypes.accumulation.size(),
        settings.vector_size_input,
    )
}

/// Shrink `plane_count` until the cube's shared footprint fits `available` bytes,
/// keeping `cube_dim.x = plane_size`, or fail if even one plane does not fit. With
/// `use_planes` there is one accumulator per plane, otherwise one per unit
/// (`plane_size * plane_count`).
fn clamp_plane_count(
    bytes_per_accumulator: usize,
    available: usize,
    plane_size: u32,
    plane_count: u32,
    use_planes: bool,
) -> Result<u32, ReduceError> {
    if bytes_per_accumulator == 0 {
        return Ok(plane_count);
    }

    let max_accumulators = available / bytes_per_accumulator;

    // The smallest cube we emit is a single plane: one accumulator in plane mode,
    // else `plane_size`. If even that does not fit the config is infeasible.
    let min_accumulators = if use_planes { 1 } else { plane_size as usize };
    if max_accumulators < min_accumulators {
        return Err(ReduceError::SharedMemoryOverflow {
            requested: bytes_per_accumulator * min_accumulators,
            available,
        });
    }

    let max_plane_count = match use_planes {
        true => max_accumulators,
        false => max_accumulators / plane_size as usize,
    };

    Ok(plane_count.min(max_plane_count as u32).max(1))
}

#[cfg(test)]
mod tests {
    use super::clamp_plane_count;
    use crate::ReduceError;

    // RTX 4090 (Ada) opt-in shared-memory limit.
    const ADA_SHARED: usize = 101_376;
    const PLANE_SIZE: u32 = 32;

    /// ArgTopK(k) over f32, no vectorization: k value + k u32 index slices of 4
    /// bytes, so 8 * k bytes per accumulator.
    fn argtopk_bytes(k: usize) -> usize {
        8 * k
    }

    #[test]
    fn clamps_argtopk_to_fit_and_leaves_sum_alone() {
        // Sum (4 bytes/accumulator) never overruns, so the wide cube is preserved.
        assert_eq!(
            clamp_plane_count(4, ADA_SHARED, PLANE_SIZE, 32, false).unwrap(),
            32
        );

        // k = 13 at width 1024 needs 8*13*1024 = 106496 > 101376, so it must shrink.
        // Across every feasible k the clamped width must fit.
        assert!(
            clamp_plane_count(argtopk_bytes(13), ADA_SHARED, PLANE_SIZE, 32, false).unwrap() < 32
        );
        for k in 1..=396 {
            let clamped =
                clamp_plane_count(argtopk_bytes(k), ADA_SHARED, PLANE_SIZE, 32, false).unwrap();
            let width = clamped as usize * PLANE_SIZE as usize;
            assert!(
                clamped >= 1 && argtopk_bytes(k) * width <= ADA_SHARED,
                "k={k}"
            );
        }
    }

    #[test]
    fn errors_when_one_warp_cannot_fit() {
        // k = 397 needs 8*397*32 = 101632 bytes for a single warp, just over the
        // limit, so even the minimum cube is infeasible.
        let err =
            clamp_plane_count(argtopk_bytes(397), ADA_SHARED, PLANE_SIZE, 32, false).unwrap_err();
        assert!(matches!(
            err,
            ReduceError::SharedMemoryOverflow { requested, available }
                if available == ADA_SHARED && requested == argtopk_bytes(397) * PLANE_SIZE as usize
        ));
    }
}
