//! [`UnitAttentionBackward`]: the unit walk owned by the queries, then the one owned by the keys.

use crate::kernels::{
    AttentionBackwardProblem, AttentionBackwardStrategy, AttentionBackwardTensors,
    UnitKeyGradientBlueprint, UnitKeyGradientRoutine, UnitKeyGradientRoutineSettings,
    UnitQueryGradientBlueprint, UnitQueryGradientRoutine, UnitQueryGradientRoutineSettings,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, UnitKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, UnitQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::Grid;

/// The backward pass on the unit routines: [`UnitQueryGradientRoutine`], then
/// [`UnitKeyGradientRoutine`]. This kind's floor: it asks nothing of the hardware, so it runs
/// everywhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitAttentionBackward;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitAttentionBackwardSettings {
    pub query_gradient: UnitQueryGradientMicrokernelSettings,
    pub key_gradient: UnitKeyGradientMicrokernelSettings,
    /// Planes each launch groups its units in, a cube.
    pub planes: NonZeroUsize,
}

impl UnitAttentionBackwardSettings {
    pub const DEFAULT: Self = Self {
        query_gradient: UnitQueryGradientMicrokernelSettings::DEFAULT,
        key_gradient: UnitKeyGradientMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for UnitAttentionBackwardSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-attention-backward query-gradient[{}] key-gradient[{}] planes={}",
            self.query_gradient, self.key_gradient, self.planes
        )
    }
}

/// What [`UnitAttentionBackward`] launches, in order.
#[derive(Debug, Clone)]
pub struct UnitAttentionBackwardBlueprint {
    pub queries: UnitQueryGradientBlueprint,
    pub keys: UnitKeyGradientBlueprint,
}

impl AttentionBackwardStrategy for UnitAttentionBackward {
    type Settings = UnitAttentionBackwardSettings;
    type Blueprint = UnitAttentionBackwardBlueprint;

    /// Each walk's microkernel's settings, the other's and the planes at their default, then a
    /// cube as full as the device holds ([`Hardware::full_cube_planes`]).
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let lead = UnitAttentionBackwardSettings::DEFAULT;
        let queries = UnitQueryGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|query_gradient| Self::Settings {
                query_gradient,
                ..lead
            });
        let keys = UnitKeyGradientMicrokernel::settings(hw)
            .into_iter()
            .skip(1)
            .map(|key_gradient| Self::Settings {
                key_gradient,
                ..lead
            });
        let full = Self::Settings {
            planes: NonZeroUsize::new(hw.full_cube_planes()).unwrap(),
            ..lead
        };

        core::iter::once(lead)
            .chain(queries)
            .chain(keys)
            .chain([full])
            .collect()
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<Self::Blueprint, Refusal> {
        Ok(UnitAttentionBackwardBlueprint {
            queries: UnitQueryGradientRoutine.prepare(
                hw,
                problem,
                &UnitQueryGradientRoutineSettings {
                    microkernel: settings.query_gradient,
                    planes: settings.planes,
                },
            )?,
            keys: UnitKeyGradientRoutine.prepare(
                hw,
                problem,
                &UnitKeyGradientRoutineSettings {
                    microkernel: settings.key_gradient,
                    planes: settings.planes,
                },
            )?,
        })
    }

    fn dispatch(&self, tensors: &AttentionBackwardTensors, blueprint: Self::Blueprint) {
        UnitQueryGradientRoutine.dispatch(tensors, &blueprint.queries);
        UnitKeyGradientRoutine.dispatch(tensors, &blueprint.keys);
    }

    fn widest_grid(blueprint: &Self::Blueprint) -> &Grid {
        &blueprint.keys.grid
    }
}
