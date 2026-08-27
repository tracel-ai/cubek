use cubecl::ir::HardwareProperties;

use crate::definition::{InterpolateError, InterpolateForwardProblem, mode_properties};

pub use cubek_tile::Residence;

/// Every choice the tile-backed interpolation launch makes.
///
/// This is the resolved end of [`InterpolateStrategy`]: nothing here is inferred any further. The
/// launch takes the geometry and the gathered input's residence exactly as stated, and only the
/// lane split is solved, by `TileGeometry::from_blueprint`, because the space asserts an exact
/// plane cover.
///
/// [`channel_block`](Self::channel_block) is the one choice inside that split a caller may still
/// pin. It is the lane's channel run, so it is the accumulator's innermost extent and sets `nr`
/// in the contraction: the separable schedule's cost is per tap, and `nr` multiplies it. Solving
/// it only ever reaches the widest divisor one line holds, which leaves the other splits of a
/// deep channel axis unreachable.
///
/// The output is always written directly to global memory. Only the gathered input can be staged,
/// so `InPlace` makes the whole tile operation in-place while `Smem` stages that input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterpolateBlueprint {
    pub input_residence: Residence,
    pub planes_per_cube: usize,
    pub rows_per_plane: usize,
    pub cols_per_lane: usize,
    /// The lane's channel run, `None` to solve it with the rest of the lane split.
    pub channel_block: Option<usize>,
}

impl InterpolateBlueprint {
    pub const fn new(
        input_residence: Residence,
        planes_per_cube: usize,
        rows_per_plane: usize,
        cols_per_lane: usize,
    ) -> Self {
        Self {
            input_residence,
            planes_per_cube,
            rows_per_plane,
            cols_per_lane,
            channel_block: None,
        }
    }

    /// Pin the lane's channel run rather than solving it. A final block may overhang the channel
    /// count; reads and writes in that padded tail are masked.
    pub const fn with_channel_block(self, block: usize) -> Self {
        Self {
            channel_block: Some(block),
            ..self
        }
    }

    pub(crate) fn validate(&self) -> Result<(), InterpolateError> {
        if self.planes_per_cube == 0 {
            return Err(InterpolateError::ZeroPlanesPerCube);
        }
        if self.rows_per_plane == 0 {
            return Err(InterpolateError::ZeroRowsPerPlane);
        }
        if self.cols_per_lane == 0 {
            return Err(InterpolateError::ZeroColsPerLane);
        }
        if self.channel_block == Some(0) {
            return Err(InterpolateError::ZeroChannelBlock);
        }
        Ok(())
    }
}

/// What a launch optimizes for.
///
/// A caller states the bottleneck it believes in and the device decides the rest:
/// [`blueprint`](Self::blueprint) reads the hardware and the problem and resolves the intent into
/// an [`InterpolateBlueprint`]. Vectorization and coalescing are not among the choices, because
/// they are not traded against anything: the lane split covers the channel axis before it rides
/// the columns, and the launch takes the widest line the device serves for the tensors it was
/// handed.
///
/// The two inferred intents are what an autotuner sweeps, and both are always launchable. On a GPU
/// they differ in how much of a cube one problem occupies and in whether the gathered input is
/// staged, which is the choice that swings a run by up to 4x either way and is therefore measured
/// rather than modelled. A CPU has neither knob, so they resolve to the same blueprint there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolateStrategy {
    /// The launch waits on memory. Widen the cube so more loads are in flight per scheduler and
    /// read the input where it lies.
    ///
    /// Always launchable: it stages nothing, so no device refuses it for want of shared memory.
    MaximizeThroughput,
    /// The launch waits on the tap window. Keep the cube at the plane count the device schedules
    /// concurrently, spend the grid instead, and stage the gathered input so a window the taps
    /// re-read is fetched once.
    ///
    /// Launchable everywhere too, but not always as asked: a device that cannot hold the window
    /// gets the in-place read instead. The window is sized from the real extents, which an
    /// autotune key only buckets, so refusing would abort on a problem the tuner never measured.
    MinimizeLatency,
    /// Pin every choice, whatever the device reports. What a characterization sweep names, and
    /// the only way to reach a channel block the lane split would not solve to.
    ///
    /// Unlike an intent, a stated `Smem` is a demand: a device that cannot hold the window refuses
    /// the launch rather than quietly reading in place, so a sweep is never told it measured a
    /// stage it did not get.
    Forced(InterpolateBlueprint),
}

/// Planes per cube for a GPU streaming one tap per output. Nothing overlaps the loads, so the
/// cube is as wide as the recorded sweeps ever wanted one.
const STREAMING_PLANES_PER_CUBE: usize = 16;

/// Planes per cube for a GPU with taps to overlap the loads with.
const WIDE_PLANES_PER_CUBE: usize = 8;

/// Planes a GPU scheduler runs concurrently: the warp schedulers on Nvidia, the SIMD units per CU
/// on AMD. The same figure the matmul selector uses.
const CONCURRENT_PLANES_PER_CUBE: usize = 4;

/// The column run a CPU lane takes. A CPU plane is one lane wide, so this is the only column
/// parallelism there is and it is what vectorizes the inner loop.
const CPU_COLS_PER_LANE: usize = 2;

/// Output rows one cube may hold live. Past this the register file is the limit rather than the
/// schedule, so the selector never proposes a deeper cube.
const MAX_ROWS_PER_CUBE: usize = 256;

impl InterpolateStrategy {
    /// The blueprint this strategy resolves to for `problem` on `hardware`.
    pub fn blueprint(
        &self,
        hardware: &HardwareProperties,
        problem: &InterpolateForwardProblem,
    ) -> InterpolateBlueprint {
        if let Self::Forced(blueprint) = self {
            return *blueprint;
        }

        // A CPU plane is one lane on one core, so neither intent has a knob to turn there: the
        // cube is the machine either way and there is nowhere to stage into. They collapse here
        // rather than at each choice below, so a sweep never measures one launch under two names.
        let is_cpu = hardware.num_cpu_cores.is_some();
        let intent = match is_cpu {
            true => Self::MaximizeThroughput,
            false => *self,
        };

        let taps = mode_properties(problem.options.mode).taps;
        let budget = rows_per_cube(hardware, problem);
        let planes = intent.planes_per_cube(hardware, taps, budget);
        let rows = rows_per_plane(problem, taps, budget / planes);

        // Lanes cover the channel axis first and ride the output columns for the rest, so a device
        // with a real plane already spreads the columns across it and a longer run per lane would
        // only hold more output live.
        let cols = match is_cpu {
            true => CPU_COLS_PER_LANE,
            false => 1,
        };

        InterpolateBlueprint::new(intent.input_residence(taps), planes, rows, cols)
    }

    /// The planes one cube holds, which is how many loads its scheduler keeps in flight.
    ///
    /// A CPU unit is a core and its plane is one lane wide, so the cube is the machine: every core
    /// takes a plane. On a GPU the count follows the intent, and a problem streaming one tap per
    /// output gives the scheduler no arithmetic to hide a load behind, so it wants the widest cube
    /// whatever the intent.
    fn planes_per_cube(&self, hardware: &HardwareProperties, taps: usize, budget: usize) -> usize {
        let wanted = match (hardware.num_cpu_cores, self) {
            (Some(cores), _) => cores as usize,
            (None, Self::MaximizeThroughput) if taps == 1 => STREAMING_PLANES_PER_CUBE,
            (None, Self::MaximizeThroughput) => WIDE_PLANES_PER_CUBE,
            (None, _) => CONCURRENT_PLANES_PER_CUBE,
        };

        // A cube wider than the device is refused at launch, and one wider than the rows it was
        // budgeted walks past the output it was meant to cover. Both are bounded here rather than
        // proposed and then paid for.
        let lanes = (hardware.plane_size_max as usize).max(1);
        let units = (hardware.max_units_per_cube as usize / lanes).max(1);

        floor_power_of_two(wanted.min(units).min(budget))
    }

    /// Where the gathered input lives.
    ///
    /// Staging pays for the window the taps re-read, so one tap has nothing to stage. Only
    /// [`MinimizeLatency`](Self::MinimizeLatency) asks for it, and it never reaches here on a CPU
    /// because the intents collapse first: the launch refuses `Smem` there outright.
    fn input_residence(&self, taps: usize) -> Residence {
        match self {
            Self::MinimizeLatency if taps > 1 => Residence::Smem,
            _ => Residence::InPlace,
        }
    }
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
    use cubecl::ir::VectorSize;

    /// A GPU with 32-lane planes, 1024 units per cube and 64 streaming multiprocessors.
    fn gpu() -> HardwareProperties {
        HardwareProperties {
            load_width: 128,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 32,
            max_shared_memory_size: 48 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_streaming_multiprocessors: Some(64),
            num_cpu_cores: None,
            num_tensor_cores: Some(4),
            min_tensor_cores_dim: Some(16),
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
            last_level_cache_size: None,
        }
    }

    /// A CPU: one lane per plane, one unit per core, no shared memory to stage into.
    fn cpu(cores: u32) -> HardwareProperties {
        HardwareProperties {
            plane_size_min: 1,
            plane_size_max: 1,
            max_units_per_cube: cores,
            num_streaming_multiprocessors: None,
            num_cpu_cores: Some(cores),
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            ..gpu()
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

    /// On a GPU the intents have to differ in what they occupy, or the autotuner measures one
    /// point twice.
    #[test]
    fn the_two_intents_resolve_to_different_blueprints() {
        let hardware = gpu();
        let problem = upsample();

        assert_ne!(
            InterpolateStrategy::MaximizeThroughput.blueprint(&hardware, &problem),
            InterpolateStrategy::MinimizeLatency.blueprint(&hardware, &problem)
        );
    }

    /// Staging is the choice that swings a run either way, so one intent has to reach it and the
    /// always-launchable one has to stay clear of it.
    #[test]
    fn only_the_latency_intent_stages_the_input() {
        let hardware = gpu();
        let problem = upsample();

        assert_eq!(
            InterpolateStrategy::MaximizeThroughput
                .blueprint(&hardware, &problem)
                .input_residence,
            Residence::InPlace
        );
        assert_eq!(
            InterpolateStrategy::MinimizeLatency
                .blueprint(&hardware, &problem)
                .input_residence,
            Residence::Smem
        );
    }

    /// One tap re-reads no window, so there is nothing for a stage to hold.
    #[test]
    fn a_single_tap_stages_nothing() {
        let problem = problem(InterpolateMode::Nearest(NearestMode::Floor), 4, 256, 512);

        assert_eq!(
            InterpolateStrategy::MinimizeLatency
                .blueprint(&gpu(), &problem)
                .input_residence,
            Residence::InPlace
        );
    }

    /// A CPU has no knob either intent can turn, so resolving them apart would leave a sweep
    /// measuring one launch under two names.
    #[test]
    fn the_intents_collapse_on_a_cpu() {
        let hardware = cpu(12);

        assert_eq!(
            InterpolateStrategy::MaximizeThroughput.blueprint(&hardware, &upsample()),
            InterpolateStrategy::MinimizeLatency.blueprint(&hardware, &upsample())
        );
    }

    /// A CPU is refused shared memory at launch, so the selector must never reach for it.
    #[test]
    fn a_cpu_never_stages_and_takes_a_plane_per_core() {
        let hardware = cpu(12);

        for strategy in [
            InterpolateStrategy::MaximizeThroughput,
            InterpolateStrategy::MinimizeLatency,
        ] {
            let blueprint = strategy.blueprint(&hardware, &upsample());

            assert_eq!(blueprint.input_residence, Residence::InPlace);
            // Twelve cores, floored to the eight a power-of-two extent reaches.
            assert_eq!(blueprint.planes_per_cube, 8);
            // A CPU plane is one lane, so the column run is the only column parallelism.
            assert!(blueprint.cols_per_lane > 1);
        }
    }

    /// Lanes already spread the columns on a device with a real plane.
    #[test]
    fn a_gpu_lane_takes_one_column() {
        assert_eq!(
            InterpolateStrategy::MaximizeThroughput
                .blueprint(&gpu(), &upsample())
                .cols_per_lane,
            1
        );
    }

    /// The row run follows the reuse a deeper run amortizes, which is the vertical resampling
    /// ratio. A downsample draws each output row from rows no other output row wants.
    #[test]
    fn the_row_run_follows_the_resampling_ratio() {
        let hardware = gpu();
        let rows = |problem| {
            InterpolateStrategy::MinimizeLatency
                .blueprint(&hardware, &problem)
                .rows_per_plane
        };

        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 64, 256)), 4);
        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 64, 64)), 1);
        assert_eq!(rows(problem(InterpolateMode::Bilinear, 4, 256, 64)), 1);
    }

    /// A cube deeper than the output it covers only masks the overhang it walked into, and one
    /// that swallows the grid leaves the device with nothing to schedule.
    #[test]
    fn a_cube_never_outruns_the_output_it_covers() {
        let hardware = gpu();

        for problem in [
            problem(InterpolateMode::Nearest(NearestMode::Floor), 1, 4, 8),
            problem(InterpolateMode::Bilinear, 1, 4, 8),
            problem(InterpolateMode::Lanczos3, 1, 8, 8),
            problem(InterpolateMode::Bicubic, 2, 32, 128),
        ] {
            for strategy in [
                InterpolateStrategy::MaximizeThroughput,
                InterpolateStrategy::MinimizeLatency,
            ] {
                let blueprint = strategy.blueprint(&hardware, &problem);
                let rows_per_cube = blueprint.planes_per_cube * blueprint.rows_per_plane;

                assert!(
                    rows_per_cube <= problem.batch * problem.output_height,
                    "{strategy:?} walks {rows_per_cube} rows of {}",
                    problem.batch * problem.output_height
                );
            }
        }
    }

    /// A cube wider than the device is refused at launch, whatever the intent asked for.
    #[test]
    fn a_narrow_device_bounds_the_plane_count() {
        let hardware = HardwareProperties {
            max_units_per_cube: 128,
            ..gpu()
        };

        let blueprint = InterpolateStrategy::MaximizeThroughput.blueprint(&hardware, &upsample());

        assert_eq!(blueprint.planes_per_cube, 128 / 32);
    }

    /// Every blueprint the selector proposes has to pass the launch's own validation, or an
    /// inferred strategy could fail a build that states nothing.
    #[test]
    fn every_inferred_blueprint_is_valid() {
        for hardware in [gpu(), cpu(1), cpu(12)] {
            for problem in [
                problem(InterpolateMode::Nearest(NearestMode::Exact), 1, 1, 1),
                problem(InterpolateMode::Bilinear, 1, 4, 8),
                problem(InterpolateMode::Lanczos3, 8, 1024, 256),
            ] {
                for strategy in [
                    InterpolateStrategy::MaximizeThroughput,
                    InterpolateStrategy::MinimizeLatency,
                ] {
                    strategy
                        .blueprint(&hardware, &problem)
                        .validate()
                        .unwrap_or_else(|e| panic!("{strategy:?} on {hardware:?}: {e}"));
                }
            }
        }
    }

    /// A stated blueprint reaches the launch untouched, whatever the device reports.
    #[test]
    fn a_forced_blueprint_is_taken_as_stated() {
        let blueprint = InterpolateBlueprint::new(Residence::Smem, 2, 8, 4).with_channel_block(3);

        assert_eq!(
            InterpolateStrategy::Forced(blueprint).blueprint(&cpu(12), &upsample()),
            blueprint
        );
    }
}
