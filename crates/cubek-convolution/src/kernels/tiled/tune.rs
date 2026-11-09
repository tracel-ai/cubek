//! How the tiled convolution is measured: the key that makes two convolutions the same problem,
//! the candidate each listed setting is raced as, and the autotune a launch without a setting runs.

use super::{
    ConvBlueprint, ConvOperation, ConvPartitioning, ConvProblem, ConvRoutine, ConvSettings,
    ConvTensors, ConvWindow, CubeStagedConvRoutine,
};
use cubecl::ir::ElemType;
use cubecl::throughput::compute_throughput_key;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeDim};
use cubecl_common::device::ServiceId;
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::cube::CubeStagedMicrokernel;
use cubek_std::autotune::{
    Autotune, CandidatePriority, Ceiling, GroupPriority, Identity, Roofline, TuneCandidate,
};
use cubek_std::routine::{EstimatedProblem, PreparedLaunches, Routine};
use cubek_std::{Hardware, Instructions, manual_mma};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What makes two convolutions want the same setting.
///
/// Every extent is **exact**: they are a model's handful of layer shapes, and anchoring would merge
/// layers whose winners differ. The batch is the axis a caller varies freely, so it is anchored to
/// the power of two above it. The widths are how wide each operand's alignment lets it be read,
/// since a setting's reads follow them.
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

    /// The widest power-of-two width `alignment` admits.
    fn width(alignment: ElemAlignment) -> usize {
        (0..=5)
            .map(|power| 1usize << power)
            .take_while(|&width| alignment.admits(width))
            .last()
            .unwrap_or(1)
    }

    /// Where a candidate launching `partitioning` in cubes of `units` units is tried: first,
    /// lowered by the doublings its boxes cover past the output's edge, the doublings its grid
    /// falls short of filling the device, the doublings the boxes read the rhs again, and, at a
    /// ring of one stage, the doublings of stages it fills one after another. Arithmetic over the
    /// key's extents, never a statement about speed.
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
        let masked = (covered / work.max(1)).max(1).ilog2();
        let cubes = boxes.iter().product::<usize>() * problem.extent(roles.batch());
        let idle = hw.cubes_to_fill(units as u32).map_or(0, |wanted| {
            (wanted as usize)
                .div_ceil(cubes.max(1))
                .next_power_of_two()
                .ilog2()
        });
        let reuse = (boxes[0] * boxes[1]).ilog2();
        let ring = match partitioning.ring().stages_in_flight.get() {
            1 => partitioning.stages().max(1).ilog2(),
            _ => 0,
        };
        CandidatePriority::FIRST.lowered(masked + idle + reuse + ring)
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
                Self::width(problem.alignments.lhs),
                Self::width(problem.alignments.rhs),
                Self::width(problem.alignments.out),
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

    /// The convolution `tensors` state, at this candidate's settings.
    ///
    /// # Errors
    ///
    /// Why the device cannot run it at these settings.
    pub fn launch(&self, tensors: ConvTensors) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let prepared =
            self.prepared
                .prepare(&CubeStagedConvRoutine, &self.hw, &problem, &self.settings)?;
        CubeStagedConvRoutine.launch_prepared(&problem, tensors, prepared)
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

/// The convolution `tensors` state, at the setting the autotune finds fastest for it among the
/// routine's listed settings on `hw`.
///
/// # Errors
///
/// The refusal of the last listed setting where none prepares the problem, so the autotune is
/// never asked to choose among declines.
pub fn launch_tuned(hw: &Hardware, tensors: ConvTensors) -> Result<(), Refusal> {
    CubeStagedConvRoutine.prepare_first_served(hw, &tensors.problem())?;
    let client = tensors.client().clone();
    let id = client.service_id();
    let hw = hw.clone();
    let tunables = TUNER.init(&id, move || candidates(hw.clone()));
    TUNER.execute(&id, &client, tunables, tensors);
    Ok(())
}

/// The autotune on `hw`: every listed setting, the lead again as the floor, a round stopping at the
/// first within reach of the instruction the convolution is bound by.
fn candidates(hw: Hardware) -> TunableSet<ConvTuneKey, ConvTensors, (), ()> {
    let mut autotune =
        Autotune::<ConvTuneKey, ConvTensors, (), ()>::new(|tensors: &ConvTensors| {
            ConvTuneKey::from(&tensors.problem())
        });
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let list = ConvCandidate::list(&hw);
    let floor = list.first().cloned();
    let roofline_hw = hw.clone();
    let autotune = autotune
        .bounded(move |key: &ConvTuneKey, tensors: &ConvTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: instruction_ceiling(&roofline_hw, key),
            work: tensors.problem().demand(),
        })
        .prioritized(&tile, list, |candidate, tensors| {
            candidate.launch(tensors).map_err(|refusal| {
                format!(
                    "{} does not serve this problem: {refusal:?}",
                    candidate.strategy()
                )
            })
        });
    let autotune = match floor {
        Some(floor) => autotune.floor(<() as Identity>::named(
            "conv-tile-floor",
            move |tensors: ConvTensors| {
                floor
                    .launch(tensors)
                    .map_err(|refusal| format!("{refusal:?}"))
            },
        )),
        None => autotune,
    };
    autotune.finish()
}

/// The instruction the convolution's arithmetic runs on: the largest the device offers at the
/// key's elements accumulating in the key's sum, plain arithmetic where it offers none.
fn instruction_ceiling(hw: &Hardware, key: &ConvTuneKey) -> Ceiling {
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
    Ceiling::Compute(compute_throughput_key(instruction, key.lhs, key.sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::ConvAlong;
    use crate::kernels::tiled::routines::test_support::{F16, cmma_8x32_only, nvidia};
    use cubek_std::autotune::priority_of;
    use std::cmp::Reverse;

    fn key(batch: usize, gathered: [usize; 2]) -> ConvTuneKey {
        ConvTuneKey::from(&ConvProblem::dense(
            batch,
            gathered,
            (64, 64),
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

    /// Seven 8-row instructions cover a map 56 wide whole, so that candidate is tried no later
    /// than the lead, whose boxes overhang it.
    #[test]
    fn a_map_56_wide_is_tried_by_the_56_row_candidate_no_later_than_the_lead() {
        let (hw, key) = (cmma_8x32_only(), key(8, [12, 58]));
        let priority = |settings| ConvCandidate::new(settings, &hw).priority(&key);
        let lead = ConvSettings::new(8, ConvAlong::new(2, 2), ConvAlong::new(2, 2));
        assert!(priority(CubeStagedConvRoutine::F7X1_P1X2_T8) >= priority(lead));
    }

    /// A map 7 wide and 7 tall is held by 8 rows, and by planes across its rows that share each
    /// weight stage.
    #[test]
    fn a_map_7_by_7_ranks_an_8_row_candidate_with_an_across_split_first() {
        let key = key(128, [9, 9]);
        let ranked = ranked(&cmma_8x32_only(), &key);
        let first = ranked[0].strategy();
        assert_eq!(rows(&ranked[0], &key), 8);
        assert!(first.grid.across.get() > 1, "{first}");
    }

    /// The same setting over a grid too small to fill the device is tried after the one over a
    /// grid that does.
    #[test]
    fn a_grid_short_of_the_device_is_tried_later() {
        let hw = cmma_8x32_only();
        let settings = CubeStagedConvRoutine::F1X1_P1X1_T8_A7;
        let priority = |key| ConvCandidate::new(settings, &hw).priority(&key);
        assert!(priority(key(1, [9, 9])) < priority(key(64, [9, 9])));
    }

    /// Boxes covering at least twice the map are tried after those that tile it.
    #[test]
    fn masked_cells_lower_the_priority() {
        let (hw, key) = (cmma_8x32_only(), key(8, [12, 58]));
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
        let many = key(8, [58, 58]);
        assert_ne!(launch(lead, &many), launch(lead.double_ring(), &many));
    }

    /// A key states the problem it describes: the key of that problem is the key.
    #[test]
    fn a_key_states_the_problem_it_describes() {
        let key = key(5, [58, 58]);
        assert_eq!(ConvTuneKey::from(&ConvProblem::from(&key)), key);
    }
}
