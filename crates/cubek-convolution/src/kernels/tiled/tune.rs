//! How the tiled convolution is measured: the key that makes two convolutions the same problem,
//! the candidate each listed setting is raced as, and the autotune a launch without a setting runs.

use super::{
    ConvBinding, ConvBlueprint, ConvOperation, ConvPartitioning, ConvProblem, ConvRoutine,
    ConvSettings, ConvTensors, ConvWindow, CubeStagedConvRoutine,
};
use cubecl::ir::ElemType;
use cubecl::throughput::{MemoryAccess, compute_throughput_key};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeDim};
use cubecl_common::device::ServiceId;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::autotune::{
    Autotune, CandidatePriority, Ceiling, GroupPriority, Identity, Roofline, TuneCandidate,
};
use cubek_std::routine::{EstimatedProblem, PreparedLaunches, ROUTINE_LISTS_NO_SETTING, Routine};
use cubek_std::{Hardware, Instructions, manual_mma};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

/// What makes two convolutions want the same setting.
///
/// Every extent is **exact**: they are a model's handful of layer shapes, and anchoring would merge
/// layers whose winners differ. The batch is the axis a caller varies freely, so it is anchored to
/// the power of two above it. The widths are how wide each operand's alignment lets it be read,
/// since a setting's reads follow them, and a view starting partway into its buffer narrows them
/// to its start's own alignment.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ConvTuneKey {
    operation: ConvOperation,
    lhs: ElemType,
    rhs: ElemType,
    out: ElemType,
    sum: ElemType,
    batch: usize,
    gathered: [usize; 2],
    out_extent: [usize; 2],
    kernel: [usize; 2],
    channels_in: usize,
    channels_out: usize,
    window: ConvWindow,
    widths: [usize; 3],
}

impl ConvTuneKey {
    /// The most batches a bucket stands for.
    const MOST_BATCH: usize = 1 << 16;

    /// The widest power-of-two width `alignment` admits for a view starting at `start`.
    fn width(alignment: ElemAlignment, start: ElemAlignment) -> usize {
        (0..=5)
            .map(|power| 1usize << power)
            .take_while(|&width| alignment.admits(width) && start.admits(width))
            .last()
            .unwrap_or(1)
    }

    /// The alignment a view of `problem`'s operand starts at, as the launch it prepares reads it.
    fn start(problem: &ConvProblem, operand: MatmulOperand) -> ElemAlignment {
        ElemAlignment::at_start(
            problem.start_bytes.get(operand),
            problem.elements.get(operand).size(),
        )
    }

    /// Where a candidate launching `partitioning` in cubes of `units` units is tried: the top band
    /// unless its boxes cover at least twice the output or its grid fills less than half the
    /// device, then one band lower. A band is raced whole and a lower one only when it measures
    /// nothing, so the priority only separates the clearly wasteful from the rest and never picks
    /// among those that could win.
    fn priority(
        &self,
        hw: &Hardware,
        partitioning: &ConvPartitioning,
        units: usize,
    ) -> CandidatePriority {
        let problem = ConvProblem::from(self);
        let roles = problem.operation.roles();
        let axes = roles.axes();
        let block = partitioning.block();
        let extents = [
            problem.extent(axes.m),
            problem.extent(roles.across()),
            problem.extent(axes.n),
        ];
        let cuts = [block.rows, block.across, block.columns];
        let boxes = [0, 1, 2].map(|axis| extents[axis].div_ceil(cuts[axis]));
        let covered: usize = (0..3).map(|axis| boxes[axis] * cuts[axis]).product();
        let work: usize = extents.iter().product();
        let wasteful = covered >= 2 * work.max(1);
        let cubes = boxes.iter().product::<usize>() * problem.extent(roles.batch());
        let idle = hw
            .cubes_to_fill(units as u32)
            .is_some_and(|wanted| wanted as usize > 2 * cubes.max(1));
        match wasteful || idle {
            true => CandidatePriority::FIRST.lowered(1),
            false => CandidatePriority::FIRST,
        }
    }
}

impl From<&ConvProblem> for ConvTuneKey {
    fn from(problem: &ConvProblem) -> Self {
        Self::new(
            problem.operation,
            problem.elements.lhs,
            problem.elements.rhs,
            problem.elements.out,
            problem.sum,
            problem.batch.next_power_of_two().min(Self::MOST_BATCH),
            problem.gathered,
            problem.out,
            problem.kernel,
            problem.channels_in,
            problem.channels_out,
            problem.window,
            [
                Self::width(
                    problem.alignments.lhs,
                    Self::start(problem, MatmulOperand::Lhs),
                ),
                Self::width(
                    problem.alignments.rhs,
                    Self::start(problem, MatmulOperand::Rhs),
                ),
                Self::width(problem.alignments.out, ElemAlignment::at_start(0, 1)),
            ],
        )
    }
}

/// The problem a key describes: dense operands of the key's elements at its widths, starting at
/// their buffers' starts.
impl From<&ConvTuneKey> for ConvProblem {
    fn from(key: &ConvTuneKey) -> Self {
        let [lhs, rhs, out] = key.widths.map(ElemAlignment::dense_rows);
        Self {
            operation: key.operation,
            batch: key.batch,
            channels_in: key.channels_in,
            gathered: key.gathered,
            channels_out: key.channels_out,
            kernel: key.kernel,
            out: key.out_extent,
            window: key.window,
            elements: MatmulOperands {
                lhs: key.lhs,
                rhs: key.rhs,
                out: key.out,
            },
            sum: key.sum,
            alignments: MatmulOperands { lhs, rhs, out },
            start_bytes: MatmulOperands::all(0),
        }
    }
}

/// One listed setting, raced for the device it was listed on.
///
/// Its priority is [`ConvTuneKey::priority`] over the launch it prepares for the key, and the launch
/// is what two settings that snap onto one are told apart by.
#[derive(Debug, Clone)]
pub struct ConvCandidate {
    settings: ConvSettings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones.
    prepared: Arc<PreparedLaunches<CubeStagedConvRoutine>>,
}

type Prepared = ConvBlueprint<CubeStagedMicrokernel>;

impl ConvCandidate {
    /// `settings` on `hw`, having prepared nothing yet.
    pub fn new(settings: ConvSettings, hw: &Hardware) -> Self {
        Self {
            settings,
            hw: hw.clone(),
            prepared: Arc::default(),
        }
    }

    /// Every setting the routine lists on `hw`, the lead first.
    pub fn list(hw: &Hardware) -> Vec<Self> {
        CubeStagedConvRoutine
            .settings(hw)
            .into_iter()
            .map(|settings| Self::new(settings, hw))
            .collect()
    }

    /// The settings this candidate launches.
    pub fn settings(&self) -> &ConvSettings {
        &self.settings
    }

    /// The convolution `tensors` state, at this candidate's settings.
    ///
    /// # Errors
    ///
    /// Why the device cannot run it at these settings, as [`serves`](Self::serves) says.
    pub fn launch(&self, tensors: ConvTensors) -> Result<(), Refusal> {
        let (problem, blueprint, binding) = self.bound(tensors)?;
        CubeStagedConvRoutine.dispatch(&problem, binding, blueprint);
        Ok(())
    }

    /// Whether the device runs `tensors` at these settings: prepared and bound, nothing launched.
    ///
    /// # Errors
    ///
    /// The refusal of the routine's preparation, or of the binding at a width its operands cannot
    /// serve.
    pub fn serves(&self, tensors: &ConvTensors) -> Result<(), Refusal> {
        self.bound(tensors.clone()).map(drop)
    }

    /// The problem `tensors` state, the launch this candidate prepares for it (each problem
    /// prepared once) and the tensors bound to it.
    fn bound(&self, tensors: ConvTensors) -> Result<(ConvProblem, Prepared, ConvBinding), Refusal> {
        let problem = tensors.problem();
        let (blueprint, grid) =
            self.prepared
                .prepare(&CubeStagedConvRoutine, &self.hw, &problem, &self.settings)?;
        let binding = tensors.bind(&problem, &blueprint, &grid)?;
        Ok((problem, blueprint, binding))
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it.
    fn prepared(&self, key: &ConvTuneKey) -> Option<(Prepared, CubeDim)> {
        let (blueprint, grid) = self
            .prepared
            .prepare(
                &CubeStagedConvRoutine,
                &self.hw,
                &ConvProblem::from(key),
                &self.settings,
            )
            .ok()?;
        Some((blueprint, grid.launch.cube_dim))
    }
}

impl TuneCandidate for ConvCandidate {
    type Strategy = ConvSettings;
    type Key = ConvTuneKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (Prepared, CubeDim);

    fn strategy(&self) -> &ConvSettings {
        &self.settings
    }

    /// Last where the routine declines the problem, which it then does at launch before compiling
    /// anything.
    fn priority(&self, key: &ConvTuneKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((blueprint, cube)) => {
                key.priority(&self.hw, &blueprint.partitioning, cube.num_elems() as usize)
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &ConvTuneKey) -> Option<Self::Blueprint> {
        self.prepared(key)
    }
}

static TUNER: LocalTuner<ConvTuneKey, ServiceId> = local_tuner!("tile-conv");

/// The list each device races, built once so every launch and the autotune share the launches its
/// candidates prepared.
static LISTS: LazyLock<Mutex<HashMap<ServiceId, Arc<[ConvCandidate]>>>> =
    LazyLock::new(Mutex::default);

fn listed(id: &ServiceId, hw: &Hardware) -> Arc<[ConvCandidate]> {
    LISTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(id.clone())
        .or_insert_with(|| ConvCandidate::list(hw).into())
        .clone()
}

/// The convolution `tensors` state, at the setting the autotune finds fastest for it among the
/// routine's listed settings on `hw`.
///
/// # Errors
///
/// The refusal of the last listed setting where none serves the problem, which is checked until
/// the autotune has settled it, so the autotune is never asked to choose among refusals.
pub fn launch_tuned(hw: &Hardware, tensors: ConvTensors) -> Result<(), Refusal> {
    let client = tensors.client().clone();
    let id = client.service_id();
    let list = listed(&id, hw);
    let tunables = TUNER.init(&id, {
        let (list, hw) = (list.clone(), hw.clone());
        move || candidates(list.clone(), hw.clone())
    });
    let key = ConvTuneKey::from(&tensors.problem());
    if TUNER.fastest_identity(&id, &tunables, &key).is_none() {
        first_served(&list, &tensors)?;
    }
    TUNER.execute(&id, &client, tunables, tensors);
    Ok(())
}

/// The first of `list` that serves `tensors`.
fn first_served(list: &[ConvCandidate], tensors: &ConvTensors) -> Result<(), Refusal> {
    let mut declined = ROUTINE_LISTS_NO_SETTING;
    for candidate in list {
        match candidate.serves(tensors) {
            Ok(()) => return Ok(()),
            Err(refusal) => declined = refusal,
        }
    }
    Err(declined)
}

/// The autotune over `list` on `hw`: every listed setting, the first of them to serve the problem
/// as the floor, a round stopping at the first within reach of the bound the convolution has.
fn candidates(
    list: Arc<[ConvCandidate]>,
    hw: Hardware,
) -> TunableSet<ConvTuneKey, ConvTensors, (), ()> {
    let mut autotune =
        Autotune::<ConvTuneKey, ConvTensors, (), ()>::new(|tensors: &ConvTensors| {
            ConvTuneKey::from(&tensors.problem())
        });
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor = list.clone();
    autotune
        .bounded(move |key: &ConvTuneKey, tensors: &ConvTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: ceiling(&hw, key),
            work: tensors.problem().demand(),
        })
        .prioritized(&tile, list.to_vec(), |candidate, tensors| {
            candidate.launch(tensors).map_err(|refusal| {
                format!(
                    "{} does not serve this problem: {refusal:?}",
                    candidate.strategy()
                )
            })
        })
        .floor(<() as Identity>::named(
            "conv-tile-floor",
            move |tensors: ConvTensors| {
                let mut declined = ROUTINE_LISTS_NO_SETTING;
                for candidate in floor.iter() {
                    match candidate.launch(tensors.clone()) {
                        Ok(()) => return Ok(()),
                        Err(refusal) => declined = refusal,
                    }
                }
                Err(format!("{declined:?}"))
            },
        ))
        .finish()
}

/// The bound the convolution has: the largest instruction the device offers at the key's elements
/// accumulating in the key's sum (plain arithmetic where it offers none), or the read bus,
/// whichever is higher for the key.
fn ceiling(hw: &Hardware, key: &ConvTuneKey) -> Ceiling {
    let instruction = hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .chain(manual_mma(hw))
        .for_triple(key.lhs, key.rhs, key.sum)
        .largest()
        .map(|shape| (shape.m, shape.n, shape.k));
    Ceiling::Either {
        compute: compute_throughput_key(instruction, key.lhs, key.sum),
        memory: MemoryAccess::Read,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::ConvAlong;
    use crate::kernels::tiled::routines::test_support::{F16, cmma_8x32_only, nvidia, problems};
    use cubek_std::autotune::priority_of;
    use std::cmp::Reverse;

    fn key(batch: usize, gathered: [usize; 2], channels_out: usize) -> ConvTuneKey {
        ConvTuneKey::from(&ConvProblem::dense(
            batch,
            gathered,
            (64, channels_out),
            [3, 3],
            1,
            F16,
        ))
    }

    /// The listed settings in the order a round tries them for `key`, the repeats left out.
    fn ranked(hw: &Hardware, key: &ConvTuneKey) -> Vec<ConvCandidate> {
        let list = ConvCandidate::list(hw);
        let mut placed: Vec<_> = (0..list.len())
            .filter_map(|position| priority_of(&list, position, key).map(|p| (p, position)))
            .collect();
        placed.sort_by_key(|&(priority, position)| (Reverse(priority), position));
        placed
            .into_iter()
            .map(|(_, position)| list[position].clone())
            .collect()
    }

    fn rows(candidate: &ConvCandidate, key: &ConvTuneKey) -> usize {
        let (blueprint, _) = candidate
            .prepared(key)
            .expect("the candidate prepares the key");
        blueprint.partitioning.block().rows
    }

    /// Every grid is wide enough to fill the device, so only the boxes' fit sets the band, and
    /// stays within the 65535 cubes a device takes along z.
    const WIDE: usize = 4096;

    /// Seven 8-row instructions cover a map 56 wide whole, and the lead's boxes overhang it
    /// little: both are raced in the top band.
    #[test]
    fn a_map_56_wide_races_the_56_row_candidate_and_the_lead_together() {
        let (hw, key) = (cmma_8x32_only(), key(WIDE, [12, 58], 128));
        let priority = |settings| ConvCandidate::new(settings, &hw).priority(&key);
        let lead = ConvSettings::new(8, ConvAlong::new(2, 2), ConvAlong::new(2, 2));
        assert_eq!(
            priority(CubeStagedConvRoutine::F7X1_P1X2_T8),
            CandidatePriority::FIRST
        );
        assert_eq!(priority(lead), CandidatePriority::FIRST);
    }

    /// A map 7 wide and 7 tall is held by 8 rows, and by planes across its rows that share each
    /// weight stage: such a candidate is raced in the top band.
    #[test]
    fn a_map_7_by_7_races_an_8_row_candidate_with_an_across_split_in_the_top_band() {
        let key = key(WIDE, [9, 9], 64);
        let top: Vec<_> = ranked(&cmma_8x32_only(), &key)
            .into_iter()
            .filter(|candidate| candidate.priority(&key) == CandidatePriority::FIRST)
            .collect();
        assert!(
            top.iter()
                .any(|c| rows(c, &key) == 8 && c.strategy().grid.across.get() > 1),
            "{:?}",
            top.iter()
                .map(|c| c.strategy().to_string())
                .collect::<Vec<_>>()
        );
    }

    /// The same setting over a grid that cannot fill half the device is raced after the one over
    /// a grid that fills it.
    #[test]
    fn a_grid_short_of_half_the_device_is_in_a_lower_band() {
        let hw = cmma_8x32_only();
        let settings = CubeStagedConvRoutine::F1X1_P1X1_T8_A7;
        let priority = |key| ConvCandidate::new(settings, &hw).priority(&key);
        assert!(priority(key(1, [9, 9], 64)) < priority(key(WIDE, [9, 9], 64)));
    }

    /// Boxes covering at least twice the map are raced after those that tile it.
    #[test]
    fn boxes_covering_twice_the_map_are_in_a_lower_band() {
        let (hw, key) = (cmma_8x32_only(), key(WIDE, [12, 58], 128));
        let priority = |settings| ConvCandidate::new(settings, &hw).priority(&key);
        let overhanging = ConvSettings::new(8, ConvAlong::new(16, 1), ConvAlong::new(1, 2));
        assert!(priority(overhanging) < priority(CubeStagedConvRoutine::F7X1_P1X2_T8));
    }

    /// A contraction of one stage compiles one launch at either ring, so the round keeps one; one
    /// of many stages keeps both.
    #[test]
    fn a_double_ring_over_one_stage_is_the_single_ring_launch() {
        let hw = nvidia();
        let launch = |settings: ConvSettings, key: &ConvTuneKey| {
            ConvCandidate::new(settings, &hw)
                .blueprint(key)
                .expect("the candidate prepares the key")
        };
        let lead = CubeStagedConvRoutine::F2X2_P2X2;
        let one_stage = ConvTuneKey::from(&ConvProblem::dense(1, [8, 8], (16, 32), [1, 1], 1, F16));
        assert_eq!(
            launch(lead, &one_stage),
            launch(lead.double_ring(), &one_stage)
        );
        let many = key(8, [58, 58], 64);
        assert_ne!(launch(lead, &many), launch(lead.double_ring(), &many));
    }

    /// The settings that win the catalog's problems are each in the round for it, none a repeat of
    /// an earlier candidate's launch.
    #[test]
    fn the_settings_that_win_are_in_the_round() {
        let hw = nvidia();
        let in_round = |settings: ConvSettings, problem: &ConvProblem| {
            let key = ConvTuneKey::from(problem);
            ranked(&hw, &key)
                .iter()
                .any(|candidate| *candidate.strategy() == settings)
        };
        let [alexnet, _, stride_2, narrow] = problems().map(|(_, problem)| problem);
        let staged = CubeStagedConvRoutine::F1X2_P2X2;
        assert!(in_round(staged.mma().double_ring(), &alexnet));
        assert!(in_round(CubeStagedConvRoutine::F2X2_P1X2, &stride_2));
        assert!(in_round(CubeStagedConvRoutine::F1X1_P1X1_T8_A7, &narrow));
    }

    /// A key states the problem it describes: the key of that problem is the key.
    #[test]
    fn a_key_states_the_problem_it_describes() {
        let key = key(5, [58, 58], 64);
        assert_eq!(ConvTuneKey::from(&ConvProblem::from(&key)), key);
    }

    /// A view starting at an odd element narrows the key's width, and the launch the key's problem
    /// prepares is the one the real problem prepares, so the candidate tuned is the one run.
    #[test]
    fn a_key_holds_the_width_a_view_start_leaves() {
        let hw = nvidia();
        let aligned = ConvProblem::dense(8, [58, 58], (64, 64), [3, 3], 1, F16);
        let mut shifted = aligned.clone();
        shifted.start_bytes.lhs = 2;
        let (aligned_key, shifted_key) = (ConvTuneKey::from(&aligned), ConvTuneKey::from(&shifted));
        assert_ne!(aligned_key, shifted_key);
        for (problem, key) in [(&aligned, &aligned_key), (&shifted, &shifted_key)] {
            for settings in ConvCandidate::list(&hw) {
                let prepared = settings.prepared.prepare(
                    &CubeStagedConvRoutine,
                    &hw,
                    problem,
                    &settings.settings,
                );
                let real = prepared
                    .ok()
                    .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim));
                assert_eq!(real, settings.blueprint(key), "{}", settings.settings);
            }
        }
    }

    /// The list and the set raced over it come out the same every time they are built.
    #[test]
    fn the_list_and_its_set_are_built_in_one_order() {
        let hw = nvidia();
        let names = |list: &[ConvCandidate]| {
            list.iter()
                .map(|candidate| candidate.settings.to_string())
                .collect::<Vec<_>>()
        };
        let (first, second) = (ConvCandidate::list(&hw), ConvCandidate::list(&hw));
        assert_eq!(names(&first), names(&second));
        let checksum =
            |list: Vec<ConvCandidate>| candidates(list.into(), hw.clone()).compute_checksum();
        assert_eq!(checksum(first), checksum(second));
    }
}
