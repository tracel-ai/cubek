use crate::components::{global::cube_dim_validation, stage::NumStages};
use crate::definition::{
    BatchMatmulBlueprint, Blueprint, CubeMappingLaunch, MatmulElems, MatmulProblem,
    MatmulSetupError, MatmulVectorSizes,
};
use crate::{args::ConfigRuntimeArg, components::batch::BatchMatmulFamily};
use crate::{
    args::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
    routines::BlueprintStrategy,
    {args::RuntimeConfig, components::CubeDimResource},
};
use cubecl::ir::HardwareProperties;
use cubecl::prelude::*;
use cubek_std::cube_count::CubeCountPlan;
use std::fmt::{Debug, Display};

/// The contract to solve a matmul
pub trait Routine<RC: RuntimeConfig>: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Debug + Clone;
}

/// The launch pipeline for matmuls with a batch matmul (might become legacy)
pub trait BatchMatmulRoutine<RC: RuntimeConfig>: Routine<RC, Blueprint: Blueprint> {
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
    ) -> Result<(), MatmulSetupError>;

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

    fn num_stages() -> NumStages;

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

    #[allow(clippy::result_large_err)]
    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}

/// Validate a blueprint against a batch-matmul family `F`. Routines delegate here from
/// their [`BatchMatmulRoutine::validate_blueprint`].
#[allow(clippy::result_large_err)]
pub fn batch_validate_blueprint<F, RC, R>(
    client: &ComputeClient<R>,
    blueprint: &F::Blueprint,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(), MatmulSetupError>
where
    RC: RuntimeConfig,
    F: BatchMatmulFamily<RC>,
    R: Runtime,
{
    F::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)
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

impl LaunchInfo<BatchMatmulBlueprint> {
    pub fn new<R: Runtime>(
        blueprint: BatchMatmulBlueprint,
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

pub(crate) fn num_concurrent_planes(properties: &HardwareProperties) -> usize {
    match properties.num_cpu_cores {
        Some(num_cores) => num_cores as usize,
        // We use the number of conccurrent planes that can work per SM at the same time on GPUs.
        //
        // This is typically the number of warp scheduler on Nvidia or the number of SIMD units
        // per CU on AMD.
        None => 4,
    }
}
