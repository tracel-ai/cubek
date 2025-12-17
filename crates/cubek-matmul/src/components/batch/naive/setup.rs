use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, server::LaunchError};

use crate::{
    components::batch::{
        BatchMatmulFamily,
        naive::{NaiveMatmul, NaiveMatmulConfig, matmul_entry},
    },
    definition::{
        CubeCountInputArgs, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem,
        MatmulSetupError,
    },
    launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}
#[derive(Debug, Clone)]
pub struct NaiveBlueprint {}

impl BatchMatmulFamily for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulPrecision> = NaiveMatmul<MP>;
    type Config = NaiveMatmulConfig;
    type Blueprint = NaiveBlueprint;

    fn expand_config<R: Runtime>(
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        _selection: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        _dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        if line_sizes.out > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Line size on output not supported",
            )));
        }

        Ok(NaiveMatmulConfig {})
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
                [*dtypes.lhs_global, *dtypes.rhs_global, *dtypes.acc_global],
                [*dtypes.lhs_stage, *dtypes.rhs_stage, *dtypes.acc_stage],
                [
                    *dtypes.lhs_register,
                    *dtypes.rhs_register,
                    *dtypes.acc_register,
                ],
            )
        }
    }
}
