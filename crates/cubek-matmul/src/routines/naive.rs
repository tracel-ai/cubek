use std::fmt::Display;

use crate::{
    components::batch::{
        BatchMatmulFamily,
        naive::{NaiveBatchMatmulFamily, NaiveBlueprint},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo, Routine},
};

pub struct NaiveRoutine {}

#[derive(Default, Clone)]
pub struct NaiveStrategy {}

impl Display for NaiveStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for NaiveStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl Routine for NaiveRoutine {
    type Strategy = NaiveStrategy;
    type BatchMatmul = NaiveBatchMatmulFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        _device_settings: &DeviceSettings<R>,
        _strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        Ok(LaunchInfo {
            blueprint: NaiveBlueprint {},
            dtypes: MatmulElems::from_globals(&problem.global_dtypes),
        })
    }

    fn can_cast_stage_element() -> bool {
        // Irrelevant
        false
    }
}
