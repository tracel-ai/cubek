use crate::{
    InterpolateError,
    definition::{InterpolateProblem, TileSize},
    routines::{
        BlueprintStrategy, GlobalInterpolateBlueprint, InterpolateBlueprint,
        InterpolateLaunchSettings, Routine, SharedMemoryBlueprint, prepare_launch_settings,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryStrategy {
    pub shared_memory_height: usize,
}

impl Routine for SharedMemoryRoutine {
    type Strategy = SharedMemoryStrategy;
    type Blueprint = SharedMemoryBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: InterpolateProblem,
        _strategy: BlueprintStrategy<Self>,
        bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let options = problem.options();
        let shared_memory_limit = client.properties().hardware.max_shared_memory_size as usize;
        let settings = prepare_launch_settings(
            client,
            &problem,
            options,
            bytes_per_element,
            vector_size,
            Some(shared_memory_limit),
        )?;

        let blueprint = InterpolateBlueprint {
            tile_size: settings.tile_size,
            options,
            global: GlobalInterpolateBlueprint::SharedMemoryBlueprint(SharedMemoryBlueprint {
                smem_width: settings.smem_width,
                smem_height: settings.smem_height,
            }),
        };

        Ok((blueprint, settings))
    }
}
