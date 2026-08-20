use cubecl::{prelude::*, tensor_vector_size_parallel};

use crate::N_VALUES_PER_THREAD;

/// Where the vectors a unit produces land in the output.
///
/// On a GPU the plane is the vector unit: neighbouring units already store one
/// contiguous stretch per step, and the parallelism is the grid. On a CPU a unit is a
/// core, the line is the only SIMD there is, and two cores must never write the same
/// cache line.
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
    #[cfg(any(test, feature = "benchmarks"))]
    Forced(PrngBlueprint),
}

/// The blueprint a launch follows and the geometry that carries it.
pub(crate) struct PrngLaunch {
    pub blueprint: PrngBlueprint,
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub line_size: usize,
    pub vectors_per_unit: u32,
}

/// The smallest run worth a threadpool wake-up: a task dispatch costs about a
/// microsecond, which at tens of GB/s of stores is tens of KB of output. Derive it
/// when `HardwareProperties` reports a dispatch cost or a cache size; it has neither today.
const MIN_BYTES_PER_UNIT: usize = 16 * 1024;

impl PrngLaunch {
    pub(crate) fn new<R: Runtime>(
        client: &ComputeClient<R>,
        output: &TensorBinding<R>,
        dtype: ElemType,
        vectors_per_draw: usize,
        strategy: PrngStrategy,
    ) -> Self {
        let blueprint = match strategy {
            #[cfg(any(test, feature = "benchmarks"))]
            PrngStrategy::Forced(blueprint) => blueprint,
            PrngStrategy::Inferred => match client.properties().hardware.num_cpu_cores {
                Some(_) => PrngBlueprint::Blocked,
                None => PrngBlueprint::Interleaved,
            },
        };

        match blueprint {
            PrngBlueprint::Interleaved => Self::interleaved(client, output.size()),
            PrngBlueprint::Blocked => Self::blocked(client, output, dtype, vectors_per_draw),
        }
    }

    fn interleaved<R: Runtime>(client: &ComputeClient<R>, size: usize) -> Self {
        let cube_dim = CubeDim::new(client, size.div_ceil(N_VALUES_PER_THREAD));
        let units = f32::ceil(size as f32 / N_VALUES_PER_THREAD as f32);
        let cubes = f32::ceil(units / cube_dim.num_elems() as f32);
        let cubes_x = f32::ceil(f32::sqrt(cubes));
        let cubes_y = f32::ceil(cubes / cubes_x);

        Self {
            blueprint: PrngBlueprint::Interleaved,
            cube_dim,
            cube_count: CubeCount::Static(cubes_x as u32, cubes_y as u32, 1),
            line_size: 1,
            vectors_per_unit: N_VALUES_PER_THREAD as u32,
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
        let min_vectors_per_unit = MIN_BYTES_PER_UNIT.div_ceil(line_size * dtype.size());
        let unit_budget = hardware
            .num_cpu_cores
            .unwrap_or(hardware.max_units_per_cube) as usize;
        let units = unit_budget
            .min(vectors.div_ceil(min_vectors_per_unit))
            .max(1);

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
            vectors_per_unit: vectors_per_unit as u32,
        }
    }
}
