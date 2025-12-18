use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, server::LaunchError};

use crate::{
    components::{
        CubeDimResource,
        batch::{
            BatchMatmulFamily,
            naive::{NaiveMatmul, NaiveMatmulConfig, matmul_entry},
        },
        global::memory::GlobalLayoutConfig,
    },
    definition::{
        Blueprint, CubeCountInputArgs, InvalidConfigError, MatmulElems, MatmulLineSizes,
        MatmulPrecision, MatmulProblem, MatmulSetupError,
    },
    launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NaiveBlueprint {}

impl Blueprint for NaiveBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }
}

impl BatchMatmulFamily for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulPrecision> = NaiveMatmul<MP>;
    type Config = NaiveMatmulConfig;
    type Blueprint = NaiveBlueprint;

    fn expand_config(blueprint: Self::Blueprint) -> Result<Self::Config, MatmulSetupError> {
        if blueprint.line_sizes.out > 1 {
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
        dtypes: &MatmulElems,
        blueprint: Self::Blueprint,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                blueprint,
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

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(8))
    }
}
