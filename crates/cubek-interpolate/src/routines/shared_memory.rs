use crate::{
    InterpolateError,
    definition::InterpolateForwardProblem,
    routines::{
        BlueprintStrategy, ForwardRoutine, GlobalInterpolateBlueprint, InterpolateBlueprint,
        InterpolateLaunchSettings, SharedMemoryBlueprint, prepare_launch_settings,
    },
};
use cubecl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedMemoryStrategy {
    pub shared_memory_height: usize,
}

impl ForwardRoutine for SharedMemoryRoutine {
    type Strategy = SharedMemoryStrategy;
    type Blueprint = SharedMemoryBlueprint;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &InterpolateForwardProblem,
        _strategy: BlueprintStrategy<Self>,
        bytes_per_element: usize,
        vector_size: usize,
    ) -> Result<(InterpolateBlueprint, InterpolateLaunchSettings), InterpolateError> {
        let options = problem.options;
        let shared_memory_limit = client.properties().hardware.max_shared_memory_size;
        let settings = prepare_launch_settings(
            client,
            problem,
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
                channel_groups: settings.channel_groups,
            }),
        };

        Ok((blueprint, settings))
    }
}
