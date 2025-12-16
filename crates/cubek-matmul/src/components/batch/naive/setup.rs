use cubecl::{
    CubeCount, CubeDim, LineSizeError, Runtime,
    client::ComputeClient,
    server::LaunchError,
    std::tensor::{launch::ViewArg, layout::Coords3d},
};

use crate::{
    components::{
        batch::{
            BatchConfig, BatchMatmul, BatchMatmulFamily,
            naive::{NaiveMatmul, NaiveMatmulConfig, matmul, matmul_entry},
        },
        global::memory::{GlobalLayout, GlobalLayoutConfig, GlobalLayoutLaunch, GlobalScaleLayout},
    },
    definition::{
        CubeCountInputArgs, HypercubeConfig, HypercubeSelection, MatmulElems, MatmulLineSizes,
        MatmulPrecision, MatmulProblem, MatmulSelection, MatmulSetupError, MatrixLayout,
    },
    launch::{InputRuntimeArg, MatmulArgs, MatmulInputHandleRef, OutputRuntimeArg},
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}
#[derive(Debug, Clone)]
pub struct NaiveBlueprint {}

impl BatchMatmulFamily for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulPrecision> = NaiveMatmul<MP>;
    type Config = NaiveMatmulConfig;
    type Blueprint = NaiveBlueprint;

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
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
