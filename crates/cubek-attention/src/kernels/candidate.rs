//! A routine of the attention family at one of its settings ([`AttentionRoutineCandidate`]), as a
//! caller that measures the routines lists, launches and identifies it.

use crate::kernels::{
    AttentionBlueprint, AttentionKey, AttentionRoutine, AttentionRoutinePrepared,
    AttentionRoutineProblem, AttentionTuneInputs, PartialAttentionRoutine, SplitKeysAttention,
};
use cubecl::tile::launch::Decline;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::{PreparedLaunches, Routine};
use std::sync::Arc;

/// A routine of the attention family at one of its settings, prepared for the device it was listed
/// for: what a caller that measures the routines races, each problem prepared once by every clone.
/// A partial routine's candidate launches it merged by a second launch
/// ([`launch_split_keys`](Self::launch_split_keys)).
#[derive(Debug, Clone)]
pub struct AttentionRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine
    /// at the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine + Copy> AttentionRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        R::settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine,
                settings,
                hw: hw.clone(),
                prepared: Arc::default(),
            })
            .collect()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }
}

impl<R: Routine> AttentionRoutineCandidate<R> {
    /// The setting this candidate launches its routine at, which names it.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionRoutineCandidate<R>
where
    R: AttentionRoutine<M, SM> + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    /// The attention `inputs` state, written into their output by this candidate's routine at its
    /// setting on the device the candidate was listed for, each problem prepared once
    /// ([`PreparedLaunches`]), or why it did not run.
    pub fn launch(&self, inputs: AttentionTuneInputs) -> Result<(), String> {
        let tensors = inputs.tensors;
        self.prepared
            .prepare(&self.hw, &tensors.problem(), &self.settings)
            .and_then(|(blueprint, grid)| self.routine.dispatch(&tensors, blueprint, grid))
            .map_err(|why| self.declines(why))
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionRoutineCandidate<R>
where
    R: PartialAttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy,
{
    /// The attention `inputs` state, written into their output by this candidate's partial
    /// routine at its setting, then the split-keys merge ([`SplitKeysAttention`]), on the device
    /// the candidate was listed for, the partial routine's launch prepared once a problem
    /// ([`PreparedLaunches`]), or why it did not run.
    pub fn launch_split_keys(&self, inputs: AttentionTuneInputs) -> Result<(), String> {
        let tensors = inputs.tensors;
        let problem = tensors.problem();
        let split = SplitKeysAttention {
            partial: self.routine,
        };
        let (blueprint, grid) = self
            .prepared
            .prepare(&self.hw, &problem, &self.settings)
            .map_err(|why| self.declines(why))?;
        let merge = split
            .merge(&self.hw, &problem, &blueprint)
            .map_err(|why| self.declines(why))?;
        split
            .dispatch(&tensors, blueprint, grid, merge)
            .map_err(|why| self.declines(why))
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionRoutineCandidate<R>
where
    R: Routine<
            Problem = AttentionRoutineProblem,
            Microkernel = M,
            Blueprint = AttentionBlueprint<M, SM>,
        >,
{
    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on the device it was listed for.
    pub fn prepared(&self, key: &AttentionKey) -> Option<AttentionRoutinePrepared<M, SM>> {
        let problem = AttentionRoutineProblem::try_from(key).ok()?;
        R::prepare(&self.hw, &problem, &self.settings)
            .ok()
            .map(|(blueprint, grid)| AttentionRoutinePrepared::new(blueprint, grid))
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionRoutineCandidate<R>
where
    R: Routine<
            Problem = AttentionRoutineProblem,
            Microkernel = M,
            Blueprint = AttentionBlueprint<M, SM>,
        >,
{
    /// What a launch reports where the routine declines its problem.
    fn declines(&self, why: Decline) -> String {
        let settings = &self.settings;
        format!("{settings} does not serve this problem: {why:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::CubeStagedAttentionRoutine;
    use crate::kernels::UnitAttentionRoutine;
    use crate::kernels::{AttentionKeyElements, AttentionVisibility, QueryPlacement, ScoreScale};
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};

    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the square `16×16×16` instruction for `f16` and for `f32`, summed in `f32`.
    fn offering(hw: Hardware) -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: 16,
            n: 16,
            k: 16,
        };
        devices::offering_cmma(hw, [square(F16), square(F32)])
    }

    /// A causal `f16` prompt of `seq` positions, `heads` query heads over `kv_heads` key heads of
    /// dim 128, one batch.
    fn key(heads: usize, kv_heads: usize, seq: usize) -> AttentionKey {
        AttentionKey::new(
            AttentionKeyElements::one(F16),
            heads,
            kv_heads,
            128,
            seq,
            seq,
            1,
            AttentionVisibility::Causal {
                queries: QueryPlacement::Last,
            },
            ScoreScale::standard(128),
            None,
            false,
            false,
            false,
            false,
        )
    }

    #[test]
    fn a_candidate_per_setting_the_default_first() {
        let hw = devices::m2_pro();
        let listed: Vec<_> = AttentionRoutineCandidate::list(CubeStagedAttentionRoutine, &hw)
            .iter()
            .map(|candidate| *candidate.settings())
            .collect();
        assert_eq!(listed, CubeStagedAttentionRoutine::settings(&hw));
    }

    /// A key whose query heads do not split over its key heads is declined by every candidate,
    /// which prepares no launch for it.
    #[test]
    fn an_uneven_group_prepares_no_launch() {
        let uneven = key(6, 4, 2048);
        for candidate in AttentionRoutineCandidate::list(UnitAttentionRoutine, &devices::m2_pro()) {
            let settings = candidate.settings();
            assert!(candidate.prepared(&uneven).is_none(), "{settings}");
        }
    }

    /// A key states the problem dense tensors of its bucket's shape do, so a candidate is placed
    /// and identified by the launch those tensors prepare.
    #[test]
    fn a_key_states_the_problem_its_dense_tensors_do() {
        let keyed = AttentionRoutineProblem::try_from(&key(32, 8, 2048)).expect("an even group");
        let extents = (keyed.seq_q, keyed.seq_kv);
        let dense = AttentionRoutineProblem {
            visibility: AttentionVisibility::Causal {
                queries: QueryPlacement::Last,
            },
            ..AttentionRoutineProblem::dense((1, 32, 8), extents, (128, 128), F16)
        };
        assert_eq!(keyed, dense);
    }

    /// On devices a dev machine cannot show: the unit routine, the family's floor, prepares
    /// everywhere, and the staged cube's lead prepares where the device offers its instruction.
    #[test]
    fn every_device_keeps_the_floor() {
        let key = key(32, 8, 2048);
        for hw in [
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
            offering(devices::h100()),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for unit in AttentionRoutineCandidate::list(UnitAttentionRoutine, &hw) {
                assert!(unit.prepared(&key).is_some(), "{}", unit.settings());
            }
        }
        let h100 = offering(devices::h100());
        let lead =
            AttentionRoutineCandidate::list(CubeStagedAttentionRoutine, &h100).swap_remove(0);
        assert!(lead.prepared(&key).is_some());
    }
}
