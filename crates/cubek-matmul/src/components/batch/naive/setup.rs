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
        Blueprint, CubeMappingLaunch, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem,
        MatmulSetupError, MatrixLayout,
    },
    launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NaiveBlueprint {
    pub line_size_out: u32,
    pub dtypes: MatmulElems,
}

impl Blueprint for NaiveBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }
}

impl BatchMatmulFamily for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulPrecision> = NaiveMatmul<MP>;
    type Config = NaiveMatmulConfig;
    type Blueprint = NaiveBlueprint;

    fn expand_config(
        _blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(NaiveMatmulConfig {})
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        cube_mapping: CubeMappingLaunch<'a, R>,
        blueprint: NaiveBlueprint,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [dtypes.lhs_stage, dtypes.rhs_stage, dtypes.acc_stage],
                [
                    dtypes.lhs_register,
                    dtypes.rhs_register,
                    dtypes.acc_register,
                ],
            )
        }
    }

    fn cubedim_resource(
        _blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        // Could be moved to blueprint to be less hard coded
        Ok(CubeDimResource::Planes(8))
    }

    fn validate_blueprint<R: Runtime>(
        _client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        _problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        _line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        if blueprint.line_size_out > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Line size on output not supported",
            )));
        }

        Ok(())
    }
}
