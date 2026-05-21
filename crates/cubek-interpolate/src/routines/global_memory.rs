use crate::{
    InterpolateError,
    definition::{InterpolateProblem, TileSize},
    routines::{
        BlueprintStrategy, GlobalInterpolateBlueprint, GlobalMemoryBlueprint, InterpolateBlueprint,
        InterpolateLaunchSettings, Routine, prepare_launch_settings,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalMemoryStrategy {}

impl Routine for GlobalMemoryRoutine {
    type Strategy = GlobalMemoryStrategy;
    type Blueprint = GlobalMemoryBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: InterpolateProblem,
        _strategy: BlueprintStrategy<Self>,
        _bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let options = problem.options();
        let settings = prepare_launch_settings(client, &problem, options, 0, vector_size, None)?;

        let blueprint = InterpolateBlueprint {
            tile_size: settings.tile_size,
            options,
            global: GlobalInterpolateBlueprint::GlobalMemoryBlueprint(GlobalMemoryBlueprint {}),
        };

        Ok((blueprint, settings))
    }
}
