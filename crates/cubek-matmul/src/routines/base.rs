use crate::components::batch::{BatchConfig, BatchMatmulFamily};
use crate::definition::{
    CubeCountInputArgs, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use cubecl::prelude::*;
use std::fmt::Debug;

/// Specifications for a matmul algorithm
pub trait Routine {
    type Strategy: Default + Debug + Clone;
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
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<Self::Blueprint, MatmulSetupError>;

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_max
    }

    // Ideally put this elsewhere
    fn can_cast_stage_element() -> bool;
}
