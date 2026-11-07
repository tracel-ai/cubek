//! Which interpolate routine runs a problem, and at what settings.

use crate::definition::{InterpolateForwardProblem, mode_properties};
use crate::kernels::{CubeInterpolateRoutineSettings, InterpolateKernel, InterpolateStrategy};
use cubecl::ir::HardwareProperties;

/// Planes a cube for a GPU streaming one tap per output. Nothing overlaps the loads, so the cube
/// is as wide as the recorded sweeps ever wanted one.
const STREAMING_PLANES_PER_CUBE: usize = 16;

/// Planes a cube for a GPU with taps to overlap the loads with.
const WIDE_PLANES_PER_CUBE: usize = 8;

/// Planes a GPU scheduler runs concurrently: the warp schedulers on Nvidia, the SIMD units per CU
/// on AMD. The same figure the matmul selector uses.
const CONCURRENT_PLANES_PER_CUBE: usize = 4;

/// The column run a CPU unit takes. A CPU plane is one unit wide, so this is the only column
/// parallelism there is and it is what vectorizes the inner loop.
const CPU_COLS_PER_UNIT: usize = 2;

/// Output rows one cube may hold live. Past this the register file is the limit rather than the
/// schedule, so the selector never proposes a deeper cube.
const MAX_ROWS_PER_CUBE: usize = 256;

/// The kernels `strategy` tries for `problem` on `hardware`, in order: the first that prepares
/// runs.
pub fn select(
    strategy: InterpolateStrategy,
    hardware: &HardwareProperties,
    problem: &InterpolateForwardProblem,
) -> Vec<InterpolateKernel> {
    let taps = mode_properties(problem.options.mode).taps;
    match strategy {
        InterpolateStrategy::Specific(kernel) => vec![kernel],
        // The cube is as wide as the loads it keeps in flight: a problem streaming one tap per
        // output gives the scheduler no arithmetic to hide a load behind.
        InterpolateStrategy::Cube => {
            let planes = match taps {
                1 => STREAMING_PLANES_PER_CUBE,
                _ => WIDE_PLANES_PER_CUBE,
            };
            vec![InterpolateKernel::Cube(settings(hardware, problem, planes))]
        }
        // One tap re-reads no window, so there is nothing for a stage to hold.
        InterpolateStrategy::CubeStaged if taps == 1 => vec![InterpolateKernel::Cube(settings(
            hardware,
            problem,
            CONCURRENT_PLANES_PER_CUBE,
        ))],
        // The window is sized from the real extents, which an autotune key only buckets: a key
        // measured staged can meet a problem whose window the device cannot hold, which then
        // reads in place at the same settings.
        InterpolateStrategy::CubeStaged => {
            let settings = settings(hardware, problem, CONCURRENT_PLANES_PER_CUBE);
            vec![
                InterpolateKernel::CubeStaged(settings),
                InterpolateKernel::Cube(settings),
            ]
        }
    }
}

/// The settings for `problem` on `hardware`, its cube `gpu_planes` planes wide on a GPU.
///
/// A CPU unit is a core and its plane is one unit wide, so the cube is the machine there: every
/// core takes a plane, and a unit takes a run of columns, the only column parallelism there is. A
/// GPU's units already spread the columns, so a longer run per unit would only hold more output
/// live.
fn settings(
    hardware: &HardwareProperties,
    problem: &InterpolateForwardProblem,
    gpu_planes: usize,
) -> CubeInterpolateRoutineSettings {
    let taps = mode_properties(problem.options.mode).taps;
    let budget = rows_per_cube(hardware, problem);
    let (wanted, cols) = match hardware.num_cpu_cores {
        Some(cores) => (cores as usize, CPU_COLS_PER_UNIT),
        None => (gpu_planes, 1),
    };
    // A cube wider than the device is refused at launch, and one wider than the rows it was
    // budgeted walks past the output it was meant to cover.
    let plane_units = (hardware.plane_size_max as usize).max(1);
    let units = (hardware.max_units_per_cube as usize / plane_units).max(1);
    let planes = floor_power_of_two(wanted.min(units).min(budget));
    let rows = rows_per_plane(problem, taps, budget / planes);
    CubeInterpolateRoutineSettings::new(planes, rows, cols)
}

/// The output rows one cube may hold live, which the plane count and the row run split between
/// them.
///
/// A cube covers `planes * rows` output rows, so a deeper cube divides the cubes there are to
/// spread over the device, and one deeper than the output itself only masks the overhang it walked
/// into. Counting the output rows alone never over-deepens a cube, since splitting the columns
/// only ever adds cubes on top.
fn rows_per_cube(hardware: &HardwareProperties, problem: &InterpolateForwardProblem) -> usize {
    // A CPU cube already holds every core, so one cube is the whole machine; a GPU spreads its
    // cubes over the streaming multiprocessors it reports.
    let cubes_wanted = hardware.num_streaming_multiprocessors.unwrap_or(1).max(1) as usize;
    let rows_available = problem.batch.saturating_mul(problem.output_height);

    (rows_available / cubes_wanted).clamp(1, MAX_ROWS_PER_CUBE)
}

/// The output rows one plane walks, within the `budget` its cube leaves it.
///
/// Consecutive output rows drawn from the same input rows re-read the window the previous row
/// already pulled, so the run follows the vertical resampling ratio: that reuse is what a deeper
/// run amortizes, and one tap per output reuses nothing.
fn rows_per_plane(problem: &InterpolateForwardProblem, taps: usize, budget: usize) -> usize {
    let reuse = match taps {
        1 => 1,
        _ => (problem.output_height / problem.input_height.max(1)).max(1),
    };

    floor_power_of_two(reuse.min(budget))
}

/// The largest power of two at most `extent`, which is at least one.
///
/// Every extent the space tiles with is a power of two: a run that is not leaves the last tile
/// part overhang, which every read and write in it then has to mask.
fn floor_power_of_two(extent: usize) -> usize {
    match extent {
        0 => 1,
        extent => 1 << extent.ilog2(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::{InterpolateMode, InterpolateOptions, NearestMode};
    use cubek_test_utils::hardware::{AVX2, GPU};

    /// `AVX2` with `cores` worker threads, each unit of a cube one thread.
    fn cpu(cores: u32) -> HardwareProperties {
        HardwareProperties {
            max_units_per_cube: cores,
            max_cube_dim: (cores, cores, cores),
            num_cpu_cores: Some(cores),
            ..AVX2
        }
    }

    fn problem(
        mode: InterpolateMode,
        batch: usize,
        input_height: usize,
        output_height: usize,
    ) -> InterpolateForwardProblem {
        InterpolateForwardProblem {
            batch,
            input_height,
            input_width: input_height,
            channels: 16,
            output_height,
            output_width: output_height,
            options: InterpolateOptions::new(mode),
        }
    }

    fn upsample() -> InterpolateForwardProblem {
        problem(InterpolateMode::Bilinear, 4, 256, 512)
    }

    /// The settings of the kernel `strategy` tries first.
    fn first(
        strategy: InterpolateStrategy,
        hardware: &HardwareProperties,
        problem: &InterpolateForwardProblem,
    ) -> InterpolateKernel {
        select(strategy, hardware, problem)[0]
    }

    fn settings_of(kernel: InterpolateKernel) -> CubeInterpolateRoutineSettings {
        match kernel {
            InterpolateKernel::Cube(settings) | InterpolateKernel::CubeStaged(settings) => settings,
        }
    }

    /// On a GPU the two strategies have to differ in what they occupy, or an autotune measures
    /// one point twice.
    #[test]
    fn the_two_strategies_take_different_settings() {
        assert_ne!(
            settings_of(first(InterpolateStrategy::Cube, &GPU, &upsample())),
            settings_of(first(InterpolateStrategy::CubeStaged, &GPU, &upsample()))
        );
    }

    /// The staged strategy stages where the window fits, and reads in place where it does not.
    #[test]
    fn the_staged_strategy_reads_in_place_where_the_window_does_not_fit() {
        let kernels = select(InterpolateStrategy::CubeStaged, &GPU, &upsample());
        let settings = settings_of(kernels[0]);

        assert_eq!(
            kernels,
            vec![
                InterpolateKernel::CubeStaged(settings),
                InterpolateKernel::Cube(settings)
            ]
        );
    }

    /// One tap re-reads no window, so there is nothing for a stage to hold.
    #[test]
    fn a_single_tap_stages_nothing() {
        let problem = problem(InterpolateMode::Nearest(NearestMode::Floor), 4, 256, 512);

        assert!(matches!(
            select(InterpolateStrategy::CubeStaged, &GPU, &problem)[..],
            [InterpolateKernel::Cube(_)]
        ));
    }

    /// A CPU has no knob either strategy can turn, so they take the same settings there.
    #[test]
    fn the_strategies_take_the_same_settings_on_a_cpu() {
        let hardware = cpu(12);

        assert_eq!(
            settings_of(first(InterpolateStrategy::Cube, &hardware, &upsample())),
            settings_of(first(InterpolateStrategy::CubeStaged, &hardware, &upsample()))
        );
    }

    /// A CPU takes a plane per core, and a unit there takes several columns.
    #[test]
    fn a_cpu_takes_a_plane_per_core() {
        let hardware = cpu(12);

        for strategy in [InterpolateStrategy::Cube, InterpolateStrategy::CubeStaged] {
            let settings = settings_of(first(strategy, &hardware, &upsample()));

            // Twelve cores, floored to the eight a power-of-two extent reaches.
            assert_eq!(settings.planes, 8);
            // A CPU plane is one unit, so the column run is the only column parallelism.
            assert!(settings.cols_per_unit > 1);
        }
    }

    /// Units already spread the columns on a device with a real plane.
    #[test]
    fn a_gpu_unit_takes_one_column() {
        assert_eq!(
            settings_of(first(InterpolateStrategy::Cube, &GPU, &upsample())).cols_per_unit,
            1
        );
    }

    /// The row run follows the reuse a deeper run amortizes, which is the vertical resampling
    /// ratio. A downsample draws each output row from rows no other output row wants.
    #[test]
    fn the_row_run_follows_the_resampling_ratio() {
        let rows = |problem| {
            settings_of(first(InterpolateStrategy::CubeStaged, &GPU, &problem)).rows_per_plane
        };

        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 64, 256)), 4);
        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 64, 64)), 1);
        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 256, 64)), 1);
    }

    /// A cube deeper than the output it covers only masks the overhang it walked into, and one
    /// that swallows the grid leaves the device with nothing to schedule.
    #[test]
    fn a_cube_never_outruns_the_output_it_covers() {
        for problem in [
            problem(InterpolateMode::Nearest(NearestMode::Floor), 1, 4, 8),
            problem(InterpolateMode::Bilinear, 1, 4, 8),
            problem(InterpolateMode::Lanczos3, 1, 8, 8),
            problem(InterpolateMode::Bicubic, 2, 32, 128),
        ] {
            for strategy in [InterpolateStrategy::Cube, InterpolateStrategy::CubeStaged] {
                let settings = settings_of(first(strategy, &GPU, &problem));
                let rows_per_cube = settings.planes * settings.rows_per_plane;

                assert!(
                    rows_per_cube <= problem.batch * problem.output_height,
                    "{strategy:?} walks {rows_per_cube} rows of {}",
                    problem.batch * problem.output_height
                );
            }
        }
    }

    /// A cube wider than the device is refused at launch, whatever the strategy asked for.
    #[test]
    fn a_narrow_device_bounds_the_plane_count() {
        let hardware = HardwareProperties {
            max_units_per_cube: 128,
            ..GPU
        };

        let settings = settings_of(first(InterpolateStrategy::Cube, &hardware, &upsample()));

        assert_eq!(settings.planes, 128 / 32);
    }

    /// A stated kernel is the only one tried, whatever the device reports.
    #[test]
    fn a_specific_kernel_is_taken_as_stated() {
        let kernel = InterpolateKernel::CubeStaged(
            CubeInterpolateRoutineSettings::new(2, 8, 4).with_channel_block(3),
        );

        assert_eq!(
            select(InterpolateStrategy::Specific(kernel), &cpu(12), &upsample()),
            vec![kernel]
        );
    }
}
