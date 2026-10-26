//! The tune candidates attention's autotunes race, the prompt's and the single position's.

use crate::kernels::tune::key::AttentionKey;
use crate::kernels::{AttentionBlueprint, AttentionRoutine, AttentionRoutineProblem};
use cubecl::CubeDim;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::short_of_the_device;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::routine::{Grid, Routine};

/// One candidate of an attention autotune: a routine of this kind at one of its
/// settings, prepared for the device the autotune is built for.
///
/// Its priority is the attention family's one term, how far the prepared launch's grid falls
/// short of filling the device ([`short_of_the_device`]).
#[derive(Debug, Clone)]
pub struct AttentionRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
}

impl<R: Routine + Copy> AttentionRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as autotune candidates, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine,
                settings,
                hw: hw.clone(),
            })
            .collect()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionRoutineCandidate<R>
where
    R: AttentionRoutine<M, SM> + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device.
    fn prepared(&self, key: &AttentionKey) -> Option<(AttentionBlueprint<M, SM>, Grid)> {
        let problem = AttentionRoutineProblem::try_from(key).ok()?;
        self.routine
            .prepare(&self.hw, &problem, &self.settings)
            .ok()
    }
}

impl<R, M: AttentionMicrokernel, SM: PartialEq> TuneCandidate for AttentionRoutineCandidate<R>
where
    R: AttentionRoutine<M, SM> + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    type Strategy = R::Settings;
    type Key = AttentionKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (AttentionBlueprint<M, SM>, CubeDim);

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &AttentionKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &AttentionKey) -> Option<Self::Blueprint> {
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::CubeStagedAttentionRoutine;
    use crate::kernels::UnitAttentionRoutine;
    use crate::kernels::{AttentionVisibility, ScoreScale};
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::output::plane::PlaneFragmentAttentionMicrokernel;
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
            F16,
            heads,
            kv_heads,
            128,
            seq,
            seq,
            1,
            AttentionVisibility::Causal,
            ScoreScale::standard(128),
            None,
        )
    }

    #[test]
    fn a_candidate_per_setting_the_default_first() {
        let hw = devices::m2_pro();
        let listed: Vec<_> = AttentionRoutineCandidate::list(
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &hw,
        )
        .iter()
        .map(|candidate| *candidate.strategy())
        .collect();
        assert_eq!(
            listed,
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new().settings(&hw)
        );
    }

    /// A key whose query heads do not split over its key heads is declined by every candidate:
    /// tried after every candidate that runs, and claiming no launch.
    #[test]
    fn an_uneven_group_is_tried_last_and_claims_no_launch() {
        let uneven = key(6, 4, 2048);
        for candidate in AttentionRoutineCandidate::list(UnitAttentionRoutine, &devices::m2_pro()) {
            let settings = candidate.strategy();
            assert_eq!(
                candidate.priority(&uneven),
                CandidatePriority::LAST,
                "{settings}"
            );
            assert_eq!(candidate.blueprint(&uneven), None, "{settings}");
        }
    }

    /// The priority reads how far the grid falls short of the device: first where a prompt's
    /// queries fill it, lower where a handful of them leave cubes idle.
    #[test]
    fn the_priority_reads_how_far_the_grid_falls_short() {
        let lead = AttentionRoutineCandidate::list(UnitAttentionRoutine, &devices::m2_pro())
            .swap_remove(0);
        assert_eq!(lead.priority(&key(32, 8, 4096)), CandidatePriority::FIRST);
        assert!(lead.priority(&key(32, 8, 1)) < CandidatePriority::FIRST);
    }

    /// A key states the problem dense tensors of its bucket's shape do, so a candidate is placed
    /// and identified by the launch those tensors prepare.
    #[test]
    fn a_key_states_the_problem_its_dense_tensors_do() {
        let keyed = AttentionRoutineProblem::try_from(&key(32, 8, 2048)).expect("an even group");
        let extents = (keyed.seq_q, keyed.seq_kv);
        let dense = AttentionRoutineProblem {
            visibility: AttentionVisibility::Causal,
            ..AttentionRoutineProblem::dense((1, 32, 8), extents, (128, 128), F16)
        };
        assert_eq!(keyed, dense);
    }

    /// On devices a dev machine cannot show: the unit routine, this kind's floor, prepares
    /// everywhere, and a cube routine's setting is tried last exactly where it claims no launch.
    #[test]
    fn every_device_keeps_the_floor_and_a_decline_claims_nothing() {
        let key = key(32, 8, 2048);
        for hw in [
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
            offering(devices::h100()),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for unit in AttentionRoutineCandidate::list(UnitAttentionRoutine, &hw) {
                assert!(unit.blueprint(&key).is_some(), "{}", unit.strategy());
            }
            for cube in AttentionRoutineCandidate::list(
                CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
                &hw,
            ) {
                let declined = cube.blueprint(&key).is_none();
                let last = cube.priority(&key) == CandidatePriority::LAST;
                assert_eq!(declined, last, "{}", cube.strategy());
            }
        }
        let h100 = offering(devices::h100());
        let lead = AttentionRoutineCandidate::list(
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &h100,
        )
        .swap_remove(0);
        assert!(lead.blueprint(&key).is_some());
    }
}
