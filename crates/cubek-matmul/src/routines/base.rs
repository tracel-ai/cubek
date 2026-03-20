use crate::components::{global::cube_dim_validation, stage::NumStages};
use crate::definition::{
    Blueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
    TilingBlueprint,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::routines::BlueprintStrategy;
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use crate::{
    components::batch::{BatchConfig, BatchMatmulFamily},
    launch::ConfigRuntimeArg,
};
use cubecl::prelude::*;
use cubek_std::cube_count::CubeCountPlan;
use std::fmt::Display;

/// Specifications for a matmul algorithm
pub trait Routine<RC: RuntimeConfig>: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Blueprint;
    type Config: BatchConfig;

    type BatchMatmul: BatchMatmulFamily<RC, Blueprint = Self::Blueprint, Config = Self::Config>;

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        config: ConfigRuntimeArg<MA, R>,
        cube_count_input: CubeMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        unsafe {
            Self::BatchMatmul::launch_unchecked::<MA, R>(
                client,
                cube_dim,
                cube_count,
                address_type,
                input,
                output,
                config,
                cube_count_input,
                blueprint,
                dtypes,
                vector_sizes,
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
        vector_sizes: MatmulVectorSizes,
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
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        Self::BatchMatmul::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)
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
    pub vector_sizes: MatmulVectorSizes,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
    pub address_type: AddressType,
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
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}

pub struct DeviceSettings<R: Runtime> {
    pub client: ComputeClient<R>,
    pub plane_dim: u32,
    pub vector_sizes: MatmulVectorSizes,
    pub max_cube_count: (u32, u32, u32),
}
