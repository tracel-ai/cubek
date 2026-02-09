use crate::components::{global::cube_dim_validation, stage::NumStages};
use crate::definition::{
    Blueprint, CubeCountPlan, CubeMappingLaunch, MatmulElems, MatmulLineSizes, MatmulProblem,
    MatmulSetupError, TilingBlueprint,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::routines::BlueprintStrategy;
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use crate::{
    components::batch::{BatchConfig, BatchMatmulFamily},
    launch::ConfigRuntimeArg,
};
use cubecl::prelude::*;
use std::fmt::Display;

/// Specifications for a matmul algorithm
pub trait Routine<RC: RuntimeConfig>: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Blueprint;
    type Config: BatchConfig;

    type BatchMatmul: BatchMatmulFamily<RC, Blueprint = Self::Blueprint, Config = Self::Config>;

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<'a, MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        config: ConfigRuntimeArg<'a, MA, R>,
        cube_count_input: CubeMappingLaunch<'a, R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        unsafe {
            Self::BatchMatmul::launch_unchecked::<MA, R>(
                client,
                cube_dim,
                cube_count,
                input,
                output,
                config,
                cube_count_input,
                blueprint,
                dtypes,
            )?
        }
        Ok(())
    }

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError>;

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError>;

    fn num_stages() -> NumStages {
        Self::BatchMatmul::num_stages()
    }

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        line_sizes: MatmulLineSizes,
    ) -> DeviceSettings<R> {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        let plane_dim = match client.properties().hardware.plane_size_max {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            line_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        Self::BatchMatmul::validate_blueprint(client, blueprint, problem, dtypes, line_sizes)
    }
}

#[derive(Debug)]
pub struct ExpandInfo<B: Blueprint> {
    pub blueprint: B,
    pub dtypes: MatmulElems,
}

#[derive(Debug)]
pub struct LaunchInfo<B: Blueprint> {
    pub blueprint: B,
    pub dtypes: MatmulElems,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
}

impl LaunchInfo<TilingBlueprint> {
    pub fn new<R: Runtime>(
        blueprint: TilingBlueprint,
        dtypes: MatmulElems,
        problem: &MatmulProblem,
        compute_resources: CubeDimResource,
        device_settings: &DeviceSettings<R>,
    ) -> Result<Self, MatmulSetupError> {
        let (cube_dim, cube_count_plan) =
            blueprint.cube_launch_info(compute_resources, problem, device_settings)?;
        cube_dim_validation(cube_dim)?;

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
        })
    }
}

pub struct DeviceSettings<R: Runtime> {
    pub client: ComputeClient<R>,
    pub plane_dim: u32,
    pub line_sizes: MatmulLineSizes,
    pub max_cube_count: (u32, u32, u32),
}
