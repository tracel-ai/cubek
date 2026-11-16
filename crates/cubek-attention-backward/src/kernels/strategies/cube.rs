//! [`CubeStagedAttentionBackward`]: the cube staged walk owned by the queries, then the one owned
//! by the keys, and a cast of the keys' sums where the key walk's heads are apart.

use crate::kernels::{
    AttentionBackwardLaunch, AttentionBackwardProblem, AttentionBackwardTensors,
    CastKeyGradientRoutine, CubeStagedKeyGradientRoutine, CubeStagedKeyGradientSumRoutine,
    CubeStagedKeyWalkBlueprint, CubeStagedKeyWalkSettings, CubeStagedQueryGradientBlueprint,
    CubeStagedQueryGradientRoutine, CubeStagedQueryGradientRoutineSettings, KeyGradientSums,
    KeyHeads, KeyHeadsBlueprint,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::key_gradient::plane::PlaneKeyGradientMicrokernelSettings;
use cubek_microkernel::attention_backward::query_gradient::cube::{
    CubeQueryGradientMicrokernel, CubeQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};
use cubek_std::routine::Grid;

/// The backward pass on the cube staged routines: [`CubeStagedQueryGradientRoutine`], then
/// [`CubeStagedKeyGradientRoutine`] with the heads of a group together, or
/// [`CubeStagedKeyGradientSumRoutine`] and [`CastKeyGradientRoutine`] with them apart. Both walks
/// hold their stages on the same ring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeStagedAttentionBackward;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedAttentionBackwardSettings {
    pub query_gradient: CubeQueryGradientMicrokernelSettings,
    pub key_gradient: CubeKeyGradientMicrokernelSettings,
    /// How a cube holds its stages.
    pub ring: StageRing,
    /// How the key walk's cubes take the heads of a group.
    pub key_heads: KeyHeads,
}

impl CubeStagedAttentionBackwardSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: CubeQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: CubeKeyGradientMicrokernelSettings::DEFAULT,
        ring: StageRing::SINGLE,
        key_heads: KeyHeads::Together,
    };
}

impl core::fmt::Display for CubeStagedAttentionBackwardSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-staged-attention-backward query-gradient[{}] key-gradient[{}]{}{}",
            self.query_gradient, self.key_gradient, self.ring, self.key_heads
        )
    }
}

/// What [`CubeStagedAttentionBackward`] launches, in order: two routines with the key walk's
/// heads together, three with them apart.
#[derive(Debug, Clone)]
pub struct CubeStagedAttentionBackwardBlueprint {
    pub queries: CubeStagedQueryGradientBlueprint,
    pub keys: KeyHeadsBlueprint<CubeStagedKeyWalkBlueprint>,
}

impl AttentionBackwardLaunch for CubeStagedAttentionBackward {
    type Settings = CubeStagedAttentionBackwardSettings;
    type Blueprint = CubeStagedAttentionBackwardBlueprint;

    /// Each walk's microkernel's settings, the other's and the ring at their default, then each
    /// ring of stages through registers, then each held key walk prefetching through registers,
    /// then both walks held on the narrow key walks, then those with the key walk's heads apart.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = CubeStagedAttentionBackwardSettings::DEFAULT;
        let queries = CubeQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = CubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let rings = [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS]
            .map(|ring| Self::Settings { ring, ..lead });
        // A key walk holding its score in fragments runs nothing but its products between two
        // fills, so its next stage lands under them: each held setting prefetching through
        // registers.
        let held = CubeKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|key_gradient| key_gradient.holds_its_score())
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ring: StageRing::SINGLE_THROUGH_REGISTERS,
                ..lead
            });
        // Both walks holding their scores in fragments, prefetching through registers: every held
        // query walk with each narrow held key walk, one key fragment a plane on two or four
        // planes. Wider key walks and a ring of one stage never won once the stages were swizzled,
        // so they are left to the lists above.
        let query_held: Vec<_> = CubeQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .filter(|query_gradient| query_gradient.holds_its_score())
            .collect();
        let narrow_keys = [(2, 2), (2, 4), (4, 2), (4, 4), (8, 4)].map(|(queries, planes)| {
            CubeKeyGradientMicrokernelSettings {
                plane: PlaneKeyGradientMicrokernelSettings::fragments(1, queries).held(),
                planes_along_keys: NonZeroUsize::new(planes).unwrap(),
            }
        });
        let both = narrow_keys
            .into_iter()
            .flat_map(|key_gradient| {
                query_held
                    .iter()
                    .map(move |&query_gradient| Self::Settings {
                        query_gradient,
                        key_gradient,
                        ring: StageRing::SINGLE_THROUGH_REGISTERS,
                        ..lead
                    })
            })
            .collect::<Vec<_>>();
        // Each of those with the key walk's heads apart: where the first keys' cube, walking every
        // head, is what the launch waits on.
        let apart = both
            .iter()
            .map(|settings| Self::Settings {
                key_heads: KeyHeads::Apart,
                ..*settings
            })
            .collect::<Vec<_>>();
        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain(rings)
            .chain(held)
            .chain(both)
            .chain(apart)
            .collect()
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<Self::Blueprint, Refusal> {
        let queries = CubeStagedQueryGradientRoutine.prepare(
            hw,
            problem,
            &CubeStagedQueryGradientRoutineSettings {
                microkernel: settings.query_gradient,
                ring: settings.ring,
            },
        )?;
        let keys = CubeStagedKeyWalkSettings {
            microkernel: settings.key_gradient,
            ring: settings.ring,
        };
        let keys = match settings.key_heads {
            KeyHeads::Together => KeyHeadsBlueprint::Together(
                CubeStagedKeyGradientRoutine.prepare(hw, problem, &keys)?,
            ),
            KeyHeads::Apart => KeyHeadsBlueprint::apart(
                CubeStagedKeyGradientSumRoutine.prepare(hw, problem, &keys)?,
            ),
        };
        Ok(CubeStagedAttentionBackwardBlueprint { queries, keys })
    }

    fn dispatch(&self, tensors: &AttentionBackwardTensors, blueprint: Self::Blueprint) {
        CubeStagedQueryGradientRoutine.dispatch(tensors, &blueprint.queries);
        match &blueprint.keys {
            KeyHeadsBlueprint::Together(walk) => {
                CubeStagedKeyGradientRoutine.dispatch(tensors, walk)
            }
            KeyHeadsBlueprint::Apart { sums: walk, cast } => {
                let sums = KeyGradientSums::zeros(tensors);
                CubeStagedKeyGradientSumRoutine.dispatch(tensors, walk, &sums);
                CastKeyGradientRoutine.dispatch(tensors, cast, &sums);
            }
        }
    }

    fn widest_grid(blueprint: &Self::Blueprint) -> &Grid {
        blueprint.keys.grid()
    }
}
