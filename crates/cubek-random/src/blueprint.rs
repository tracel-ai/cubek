use cubecl::{prelude::*, tensor_vector_size_parallel};
use cubek_std::cube_count::cube_count_spread_with_total;

use crate::N_VALUES_PER_THREAD;

/// The device a launch is built for, named by where the vectors a unit produces land in
/// the output.
///
/// On a GPU the plane is the vector unit: neighbouring units already store one
/// contiguous stretch per step, and the parallelism is the grid. On a CPU a unit is a
/// core, the line is the only SIMD there is, and two cores must never write the same
/// cache line.
///
/// A draw reads the same choice for what its device does cheaply: `Interleaved` has
/// transcendentals in hardware and `Blocked` has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PrngBlueprint {
    /// Neighbouring units own neighbouring vectors, and each unit writes a fixed
    /// number of them.
    Interleaved,
    /// Each unit owns one contiguous run of the output, as long as the tensor makes it.
    Blocked,
}

/// Whether a launch derives its geometry or runs a pinned one.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PrngStrategy {
    Inferred,
    /// Pins the blueprint whatever the device reports, so an A/B of the two runs in
    /// one process.
    ///
    /// Only tests and benchmarks pin one, so a default build sees nobody construct it.
    #[allow(dead_code)]
    Forced(PrngBlueprint),
}

/// The blueprint a launch follows and the geometry that carries it.
pub(crate) struct PrngLaunchSettings {
    pub blueprint: PrngBlueprint,
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub line_size: usize,
    pub vectors_per_unit: u32,
}

impl PrngLaunchSettings {
    pub(crate) fn new<R: Runtime>(
        client: &ComputeClient<R>,
        output: &TensorBinding<R>,
        dtype: ElemType,
        vectors_per_draw: usize,
        strategy: PrngStrategy,
    ) -> Self {
        let blueprint = match strategy {
            PrngStrategy::Forced(blueprint) => blueprint,
            PrngStrategy::Inferred => match client.properties().hardware.num_cpu_cores {
                Some(_) => PrngBlueprint::Blocked,
                None => PrngBlueprint::Interleaved,
            },
        };

        match blueprint {
            PrngBlueprint::Interleaved => Self::interleaved(client, output, dtype),
            PrngBlueprint::Blocked => Self::blocked(client, output, dtype, vectors_per_draw),
        }
    }

    fn interleaved<R: Runtime>(
        client: &ComputeClient<R>,
        output: &TensorBinding<R>,
        dtype: ElemType,
    ) -> Self {
        let size = output.size();

        // Every lane already draws its own decorrelated stream (see `PrngState::seeded`),
        // so nothing but the output layout bounds the line.
        let line_size = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(dtype.size()),
            &output.shape,
            &output.strides,
            output.strides.len() - 1,
        );

        let units = size.div_ceil(N_VALUES_PER_THREAD);
        let cube_dim = CubeDim::new(client, units);
        let cubes = units.div_ceil(cube_dim.num_elems() as usize);
        let (cube_count, _) = cube_count_spread_with_total(client, cubes);

        Self {
            blueprint: PrngBlueprint::Interleaved,
            cube_dim,
            cube_count,
            line_size,
            vectors_per_unit: (N_VALUES_PER_THREAD / line_size) as u32,
        }
    }

    fn blocked<R: Runtime>(
        client: &ComputeClient<R>,
        output: &TensorBinding<R>,
        dtype: ElemType,
        vectors_per_draw: usize,
    ) -> Self {
        let hardware = &client.properties().hardware;

        // The state, not the output element, sets the line: four `u32` registers advance
        // per draw against a single store, so the widest line worth building is the one a
        // register holds the state at.
        let line_size = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(size_of::<u32>()),
            &output.shape,
            &output.strides,
            output.strides.len() - 1,
        );

        let vectors = output.size().div_ceil(line_size);

        // The smallest run worth a threadpool wake-up: a dispatch costs about a
        // microsecond, which at tens of GB/s of stores is tens of KB. Half the L1d,
        // which the CPU runtime reports as its shared-memory analogue, lands there.
        let min_bytes_per_unit = hardware.max_shared_memory_size / 2;
        let min_vectors_per_unit = min_bytes_per_unit.div_ceil(line_size * dtype.size()).max(1);
        let unit_budget = hardware
            .num_cpu_cores
            .unwrap_or(hardware.max_units_per_cube) as usize;
        let units = unit_budget.min(vectors / min_vectors_per_unit).max(1);

        // A run covers whole draws, and the last unit's may reach past the output; the
        // write is checked, so the overshoot is dropped rather than clamped. An empty
        // tensor still gets a run, which the same check empties.
        let vectors_per_unit = (vectors.div_ceil(units).div_ceil(vectors_per_draw)
            * vectors_per_draw)
            .max(vectors_per_draw);
        let units = vectors.div_ceil(vectors_per_unit).max(1);

        let units_per_cube = units.min(hardware.max_units_per_cube as usize);

        Self {
            blueprint: PrngBlueprint::Blocked,
            cube_dim: CubeDim::new_1d(units_per_cube as u32),
            cube_count: CubeCount::Static(units.div_ceil(units_per_cube) as u32, 1, 1),
            line_size,
            vectors_per_unit: u32::try_from(vectors_per_unit)
                .unwrap_or_else(|_| panic!("vectors_per_unit {vectors_per_unit} overflows u32")),
        }
    }
}

#[cfg(test)]
mod tests {
    use cubecl::{TestRuntime, std::tensor::TensorHandle};

    use super::*;

    /// `PrngStrategy::Inferred` reads the hardware's core count to pick a blueprint, and
    /// nothing else checks that the two branches of the mapping stay in sync with it.
    #[test]
    fn inferred_maps_num_cpu_cores_to_blocked() {
        let client = TestRuntime::client(&Default::default());
        let dtype = f32::elem_type_native();

        let expected = if client.properties().hardware.num_cpu_cores.is_some() {
            PrngBlueprint::Blocked
        } else {
            PrngBlueprint::Interleaved
        };

        let output = TensorHandle::<TestRuntime>::empty(&client, vec![64, 64], dtype);
        let settings =
            PrngLaunchSettings::new(&client, &output.binding(), dtype, 1, PrngStrategy::Inferred);

        assert_eq!(settings.blueprint, expected);
    }
}
