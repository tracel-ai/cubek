//! [`HypercubeAttentionBackward`]: the delta routine, the hypercube walk, and the casts out of the
//! sums it added into.

use crate::kernels::{
    AttentionBackwardLaunch, AttentionBackwardProblem, AttentionBackwardTensors,
    CastKeyGradientRoutine, CastQueryGradientBlueprint, CastQueryGradientRoutine,
    HypercubeGradientSumRoutine, HypercubeKeyGradientQuerySumRoutine, HypercubeKeyWalkBlueprint,
    HypercubeKeyWalkSettings, KeyGradientSums, KeyHeads, KeyHeadsBlueprint, PlaneDeltaBlueprint,
    PlaneDeltaRoutine, PlaneDeltaRoutineSettings, QueryGradientSum,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::delta::plane::PlaneDeltaMicrokernelSettings;
use cubek_microkernel::attention_backward::key_gradient::hypercube::{
    HypercubeKeyGradientMicrokernel, HypercubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};
use cubek_std::routine::Grid;

/// The backward pass that walks the tiles once, owned by the keys: [`PlaneDeltaRoutine`], then
/// [`HypercubeKeyGradientQuerySumRoutine`] with the heads of a group together, or
/// [`HypercubeGradientSumRoutine`] and [`CastKeyGradientRoutine`] with them apart, then
/// [`CastQueryGradientRoutine`] under the delta routine's planes. A device without `f32` atomic
/// adds declines it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HypercubeAttentionBackward;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAttentionBackwardSettings {
    pub delta: PlaneDeltaMicrokernelSettings,
    pub key_gradient: HypercubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
    /// How the key walk's cubes take the heads of a group.
    pub key_heads: KeyHeads,
}

impl HypercubeAttentionBackwardSettings {
    pub const DEFAULT: Self = Self {
        delta: PlaneDeltaMicrokernelSettings::DEFAULT,
        key_gradient: HypercubeKeyGradientMicrokernelSettings::DEFAULT,
        ring: StageRing::SINGLE,
        key_heads: KeyHeads::Together,
    };
}

impl core::fmt::Display for HypercubeAttentionBackwardSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-attention-backward delta[{}] key-gradient[{}]{}{}",
            self.delta, self.key_gradient, self.ring, self.key_heads
        )
    }
}

/// What [`HypercubeAttentionBackward`] launches, in order: three routines with the key walk's
/// heads together, four with them apart.
#[derive(Debug, Clone)]
pub struct HypercubeAttentionBackwardBlueprint {
    pub delta: PlaneDeltaBlueprint,
    pub keys: KeyHeadsBlueprint<HypercubeKeyWalkBlueprint>,
    pub cast_queries: CastQueryGradientBlueprint,
}

impl AttentionBackwardLaunch for HypercubeAttentionBackward {
    type Settings = HypercubeAttentionBackwardSettings;
    type Blueprint = HypercubeAttentionBackwardBlueprint;

    /// The key walk's microkernel's settings at the default ring, then each ring of stages through
    /// registers, then each held key walk on each of them, then some of those with the key walk's
    /// heads apart.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = HypercubeAttentionBackwardSettings::DEFAULT;
        let keys = HypercubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let rings = [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS]
            .map(|ring| Self::Settings { ring, ..lead });
        // A key walk holding its score in fragments runs nothing but its products between two
        // fills, so its next stage lands under them: each held setting on each ring.
        let held = HypercubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|key_gradient| key_gradient.holds_its_score())
            .flat_map(|key_gradient| {
                [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS].map(|ring| {
                    Self::Settings {
                        key_gradient,
                        ring,
                        ..lead
                    }
                })
            })
            .collect::<Vec<_>>();
        // The key walk's heads apart, on the held settings prefetching through registers: a
        // launch of few cubes, each holding a whole group, leaves most of the device idle while
        // the last ones run.
        let apart = held
            .iter()
            .filter(|settings| settings.ring == StageRing::SINGLE_THROUGH_REGISTERS)
            .map(|settings| Self::Settings {
                key_heads: KeyHeads::Apart,
                ..*settings
            })
            .collect::<Vec<_>>();
        core::iter::once(lead)
            .chain(keys)
            .chain(rings)
            .chain(held)
            .chain(apart)
            .collect()
    }

    /// # Errors
    ///
    /// [`ATTENTION_BACKWARD_NO_ATOMIC_ADD`](crate::kernels::ATTENTION_BACKWARD_NO_ATOMIC_ADD) on a
    /// device without `f32` atomic adds, and what any routine declines.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<Self::Blueprint, Refusal> {
        let keys = HypercubeKeyWalkSettings {
            microkernel: settings.key_gradient,
            ring: settings.ring,
        };
        // The key walk first: it is what declines a device without atomic adds.
        let keys = match settings.key_heads {
            KeyHeads::Together => KeyHeadsBlueprint::Together(
                HypercubeKeyGradientQuerySumRoutine.prepare(hw, problem, &keys)?,
            ),
            KeyHeads::Apart => {
                KeyHeadsBlueprint::apart(HypercubeGradientSumRoutine.prepare(hw, problem, &keys)?)
            }
        };
        let delta = PlaneDeltaRoutine.prepare(
            hw,
            problem,
            &PlaneDeltaRoutineSettings {
                microkernel: settings.delta,
            },
        )?;
        Ok(HypercubeAttentionBackwardBlueprint {
            cast_queries: CastQueryGradientRoutine.prepare(hw, &delta),
            delta,
            keys,
        })
    }

    fn dispatch(&self, tensors: &AttentionBackwardTensors, blueprint: Self::Blueprint) {
        let query_sum = QueryGradientSum::zeros(tensors);
        PlaneDeltaRoutine.dispatch(tensors, &blueprint.delta);
        match &blueprint.keys {
            KeyHeadsBlueprint::Together(walk) => {
                HypercubeKeyGradientQuerySumRoutine.dispatch(tensors, walk, &query_sum)
            }
            KeyHeadsBlueprint::Apart { sums: walk, cast } => {
                let sums = KeyGradientSums::zeros(tensors);
                HypercubeGradientSumRoutine.dispatch(tensors, walk, &sums, &query_sum);
                CastKeyGradientRoutine.dispatch(tensors, cast, &sums);
            }
        }
        CastQueryGradientRoutine.dispatch(tensors, &blueprint.cast_queries, &query_sum);
    }

    fn widest_grid(blueprint: &Self::Blueprint) -> &Grid {
        blueprint.keys.grid()
    }
}
