//! [`PlaneFragmentAttentionBackward`]: the plane walk owned by the queries, then the one owned by
//! the keys.

use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardStrategy, AttentionBackwardTensors,
    PlaneFragmentKeyGradientBlueprint, PlaneFragmentKeyGradientRoutine,
    PlaneFragmentKeyGradientRoutineSettings, PlaneFragmentQueryGradientBlueprint,
    PlaneFragmentQueryGradientRoutine, PlaneFragmentQueryGradientRoutineSettings,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, PlaneKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};
use cubek_std::routine::Grid;

/// The backward pass on the plane fragment routines: [`PlaneFragmentQueryGradientRoutine`], then
/// [`PlaneFragmentKeyGradientRoutine`], both on the same planes a cube and the same ring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneFragmentAttentionBackward;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneFragmentAttentionBackwardSettings {
    pub query_gradient: PlaneQueryGradientMicrokernelSettings,
    pub key_gradient: PlaneKeyGradientMicrokernelSettings,
    /// Planes each launch groups in one cube.
    pub planes: NonZeroUsize,
    /// How a plane holds its stages.
    pub ring: StageRing,
}

impl PlaneFragmentAttentionBackwardSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: PlaneQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: PlaneKeyGradientMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
    };
}

impl core::fmt::Display for PlaneFragmentAttentionBackwardSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-fragment-attention-backward query-gradient[{}] key-gradient[{}] planes={}{}",
            self.query_gradient, self.key_gradient, self.planes, self.ring
        )
    }
}

/// What [`PlaneFragmentAttentionBackward`] launches, in order.
#[derive(Debug, Clone)]
pub struct PlaneFragmentAttentionBackwardBlueprint {
    pub queries: PlaneFragmentQueryGradientBlueprint,
    pub keys: PlaneFragmentKeyGradientBlueprint,
}

impl AttentionBackwardStrategy for PlaneFragmentAttentionBackward {
    type Settings = PlaneFragmentAttentionBackwardSettings;
    type Blueprint = PlaneFragmentAttentionBackwardBlueprint;

    /// Each walk's microkernel's settings, the other's and the strategy's at their default, then
    /// two planes a cube, and each ring of stages through registers.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = PlaneFragmentAttentionBackwardSettings::DEFAULT;
        let queries = PlaneQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = PlaneKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let rings = [StageRing::DOUBLE, StageRing::SINGLE_THROUGH_REGISTERS]
            .map(|ring| Self::Settings { ring, ..lead });
        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain([Self::Settings {
                planes: NonZeroUsize::new(2).unwrap(),
                ..lead
            }])
            .chain(rings)
            .collect()
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<Self::Blueprint, Refusal> {
        Ok(PlaneFragmentAttentionBackwardBlueprint {
            queries: PlaneFragmentQueryGradientRoutine.prepare(
                hw,
                problem,
                &PlaneFragmentQueryGradientRoutineSettings {
                    microkernel: settings.query_gradient,
                    planes: settings.planes,
                    ring: settings.ring,
                },
            )?,
            keys: PlaneFragmentKeyGradientRoutine.prepare(
                hw,
                problem,
                &PlaneFragmentKeyGradientRoutineSettings {
                    microkernel: settings.key_gradient,
                    planes: settings.planes,
                    ring: settings.ring,
                },
            )?,
        })
    }

    fn dispatch(&self, tensors: &AttentionBackwardTensors, blueprint: Self::Blueprint) {
        PlaneFragmentQueryGradientRoutine.dispatch(tensors, &blueprint.queries);
        PlaneFragmentKeyGradientRoutine.dispatch(tensors, &blueprint.keys);
    }

    fn widest_grid(blueprint: &Self::Blueprint) -> &Grid {
        &blueprint.keys.grid
    }
}
