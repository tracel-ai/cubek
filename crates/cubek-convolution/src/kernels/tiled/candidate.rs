use crate::kernels::tiled::{
    ConvProblem, ConvRoutine, ConvTensors, CubeStagedConvPrepared, CubeStagedConvRoutine,
    CubeStagedConvSettings, CubeStagedConvTuneKey,
};
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;
use cubek_std::routine::RoutineCandidate;

/// One setting the cube staged routine lists, for the device it was listed on: what a caller that
/// measures the settings identifies a candidate by, and launches again once it is found fastest.
///
/// # Guarantees
///
/// Clones share the launches the candidate prepared, each problem prepared once.
#[derive(Debug, Clone)]
pub struct CubeStagedConvCandidate {
    candidate: RoutineCandidate<CubeStagedConvRoutine>,
}

impl CubeStagedConvCandidate {
    /// `settings` on `hw`, having prepared nothing yet.
    pub fn new(settings: CubeStagedConvSettings, hw: &Hardware) -> Self {
        Self {
            candidate: RoutineCandidate::new(CubeStagedConvRoutine, settings, hw),
        }
    }

    /// Every setting the routine lists on `hw`, the first of them the base.
    ///
    /// # Guarantees
    ///
    /// The same settings in the same order for the same `hw`, so a set raced over them checksums
    /// alike.
    pub fn list(hw: &Hardware) -> Vec<Self> {
        RoutineCandidate::list(CubeStagedConvRoutine, hw)
            .into_iter()
            .map(|candidate| Self { candidate })
            .collect()
    }

    /// The settings this candidate launches, which name it.
    pub fn settings(&self) -> &CubeStagedConvSettings {
        self.candidate.settings()
    }

    /// The convolution `tensors` state, at this candidate's settings.
    ///
    /// # Errors
    ///
    /// Why the device cannot run it at these settings, as [`serves`](Self::serves) says.
    pub fn launch(&self, tensors: ConvTensors) -> Result<(), Decline> {
        let problem = tensors.problem();
        let prepared = self.candidate.prepare(&problem)?;
        CubeStagedConvRoutine.launch_prepared(&problem, tensors, prepared)
    }

    /// Whether the device runs `tensors` at these settings: prepared and bound, nothing launched.
    ///
    /// # Errors
    ///
    /// The refusal of the routine's preparation, or of the binding at a width its operands cannot
    /// serve.
    pub fn serves(&self, tensors: &ConvTensors) -> Result<(), Decline> {
        let problem = tensors.problem();
        let (blueprint, grid) = self.candidate.prepare(&problem)?;
        tensors.clone().bind(&problem, &blueprint, &grid).map(drop)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it.
    ///
    /// # Guarantees
    ///
    /// - The launch is the one [`launch`](Self::launch) compiles for every convolution `key`
    ///   keys.
    /// - Where it is `None`, [`launch`](Self::launch) declines before compiling anything.
    pub fn prepared(&self, key: &CubeStagedConvTuneKey) -> Option<CubeStagedConvPrepared> {
        let problem = ConvProblem::from(key);
        let (blueprint, cube_dim) = self.candidate.launch_of(&problem)?;
        Some(CubeStagedConvPrepared::new(&problem, blueprint, cube_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::benchmarks::{Conv2dElement, Conv2dProblem, settings};
    use crate::kernels::tiled::routines::PlaneWindowConvRoutine;
    use crate::kernels::tiled::routines::test_support::{cmma_8x32_only, dense, nvidia};
    use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
    use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings as Fragments;
    use cubek_std::devices::topology;
    use cubek_std::routine::Routine;

    const ROWS_8: usize = 8;
    const FRAGMENTS_16: usize = 16;

    /// Every grid is wide enough to fill the device, so only the boxes' fit varies, and stays
    /// within the 65535 cubes a device takes along z.
    const WIDE: usize = 4096;

    fn key(problem: Conv2dProblem) -> CubeStagedConvTuneKey {
        CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&problem), &nvidia())
    }

    /// `WIDE` batches of a map 56 wide, held by a few rows.
    fn wide_map_56() -> Conv2dProblem {
        Conv2dProblem {
            input_shape: [WIDE, 64, 12, 58],
            weight_shape: [128, 64, 3, 3],
            ..dense()
        }
    }

    /// `WIDE` batches of a map 7 wide and 7 tall.
    fn wide_map_7() -> Conv2dProblem {
        Conv2dProblem {
            input_shape: [WIDE, 64, 9, 9],
            ..dense()
        }
    }

    /// The launch `settings` prepare on `hw` for `key`.
    fn prepared(
        settings: CubeStagedConvSettings,
        hw: &Hardware,
        key: &CubeStagedConvTuneKey,
    ) -> CubeStagedConvPrepared {
        CubeStagedConvCandidate::new(settings, hw)
            .prepared(key)
            .expect("the candidate prepares the key")
    }

    /// Seven 8-row instructions cover a map 56 wide whole, and the base's boxes overhang it
    /// little: neither covers twice the map.
    #[test]
    fn a_map_56_wide_is_covered_closely_by_the_56_row_candidate_and_the_base() {
        let (hw, key) = (cmma_8x32_only(), key(wide_map_56()));
        let base = CubeStagedConvSettings::new(
            ROWS_8,
            Fragments::grid(2, 2, 1),
            MatmulPlaneLayout::new(2, 2),
        );
        for settings in [settings::F7X1_P1X2_T8, base] {
            let coverage = *prepared(settings, &hw, &key).coverage();
            assert!(
                coverage.covered_cells < 2 * coverage.problem_cells,
                "{settings}"
            );
        }
    }

    /// Boxes of sixteen fragments of rows overhang a map 56 wide by at least twice its cells.
    #[test]
    fn boxes_of_sixteen_fragments_cover_twice_a_map_56_wide() {
        let (hw, key) = (cmma_8x32_only(), key(wide_map_56()));
        let overhanging = CubeStagedConvSettings::new(
            ROWS_8,
            Fragments::grid(FRAGMENTS_16, 1, 1),
            MatmulPlaneLayout::new(1, 2),
        );
        let coverage = *prepared(overhanging, &hw, &key).coverage();
        assert!(coverage.covered_cells >= 2 * coverage.problem_cells);
    }

    /// One batch of a map 7 wide spawns a cube for each box of its one position across, so its
    /// grid falls short of half what fills the device, where `WIDE` batches fill it.
    #[test]
    fn one_batch_of_a_map_7_wide_spawns_short_of_half_the_device() {
        let hw = cmma_8x32_only();
        let settings = settings::F1X1_P1X1_T8_A7;
        let short = |problem| {
            let prepared = prepared(settings, &hw, &key(problem));
            let wanted = hw
                .cubes_to_fill(prepared.cube_dim().num_elems())
                .expect("the device states its cores");
            wanted as usize > 2 * prepared.coverage().cube_count
        };
        assert!(short(Conv2dProblem {
            input_shape: [1, 64, 9, 9],
            ..wide_map_7()
        }));
        assert!(!short(wide_map_7()));
    }

    /// A contraction of one stage compiles one launch at either ring, so the round keeps one; one
    /// of many stages keeps both.
    #[test]
    fn a_double_ring_over_one_stage_is_the_single_ring_launch() {
        let hw = nvidia();
        let base = settings::F2X2_P2X2_T16;
        let one_stage = key(Conv2dProblem {
            input_shape: [1, 16, 8, 8],
            weight_shape: [32, 16, 1, 1],
            ..dense()
        });
        assert_eq!(
            prepared(base, &hw, &one_stage),
            prepared(base.double_ring(), &hw, &one_stage)
        );
        let many = key(dense());
        assert_ne!(
            prepared(base, &hw, &many),
            prepared(base.double_ring(), &hw, &many)
        );
    }

    /// The list comes out the same every time it is built, so an autotune over it checksums
    /// alike.
    #[test]
    fn the_list_is_built_in_one_order() {
        let hw = nvidia();
        let names = || {
            CubeStagedConvCandidate::list(&hw)
                .iter()
                .map(|candidate| candidate.settings().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(), names());
    }

    /// Whether the plane window routine serves `problem` on `hw`.
    fn plane_window_serves(hw: &Hardware, problem: &Conv2dProblem) -> bool {
        PlaneWindowConvRoutine::prepare_first_served(hw, &ConvProblem::from(problem)).is_ok()
    }

    /// A device whose plane width is a range lists nothing, and the plane window routine still
    /// serves it.
    #[test]
    fn a_ranged_plane_width_lists_nothing_the_plane_window_routine_serves() {
        let mut hw = nvidia();
        let device = topology(&mut hw);
        (device.plane_size_min, device.plane_size_max) = (32, 64);
        assert!(CubeStagedConvCandidate::list(&hw).is_empty());
        assert!(plane_window_serves(&hw, &dense()));
    }

    /// An element the device has no matrix instruction for leaves every listed setting declining
    /// the key, and the plane window routine still serves it.
    #[test]
    fn an_f32_problem_without_f32_instructions_is_served_by_the_plane_window_routine_alone() {
        let hw = nvidia();
        let problem = Conv2dProblem {
            element: Conv2dElement::F32,
            ..dense()
        };
        let key = key(problem.clone());
        let list = CubeStagedConvCandidate::list(&hw);
        assert!(!list.is_empty());
        assert!(
            list.iter()
                .all(|candidate| candidate.prepared(&key).is_none())
        );
        assert!(plane_window_serves(&hw, &problem));
    }
}
