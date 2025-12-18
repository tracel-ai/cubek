use crate::components::batch::{BatchConfig, BatchMatmulFamily};
use crate::definition::{
    CubeCountInputArgs, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::routines::BlueprintStrategy;
use cubecl::prelude::*;
use std::fmt::{Debug, Display};

/// Specifications for a matmul algorithm
pub trait Routine: Sized {
    type Strategy: Default + Display + Clone;
    type Blueprint: Debug + Clone;
    type Config: BatchConfig;

    type BatchMatmul: BatchMatmulFamily<Blueprint = Self::Blueprint, Config = Self::Config>;

    fn expand_config<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<<Self::BatchMatmul as BatchMatmulFamily>::Config, MatmulSetupError> {
        Self::BatchMatmul::expand_config(client, problem, selection, line_sizes, dtypes)
    }

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: <Self::BatchMatmul as BatchMatmulFamily>::Config,
        dtypes: &MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        match unsafe {
            Self::BatchMatmul::launch_unchecked::<MA, R>(
                client,
                cube_dim,
                cube_count,
                input,
                output,
                cube_count_input,
                config,
                dtypes,
            )
        } {
            Ok(_) => Ok(()),
            Err(err) => Err(MatmulSetupError::Launch(err)),
        }
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError>;

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_max
    }

    // Ideally put this elsewhere
    fn can_cast_stage_element() -> bool;
}

pub struct LaunchInfo<B: Debug + Clone> {
    pub blueprint: B,
    pub dtypes: MatmulElems,
}

pub struct DeviceSettings<R: Runtime> {
    pub client: ComputeClient<R>,
    pub plane_dim: u32,
    pub line_sizes: MatmulLineSizes,
}
