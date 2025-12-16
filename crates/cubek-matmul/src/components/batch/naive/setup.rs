use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    server::LaunchError,
    std::tensor::{launch::ViewArg, layout::Coords3d},
};

use crate::{
    components::{
        batch::{
            BatchConfig, BatchMatmulFamily, CubeCountInputArgs,
            naive::{NaiveMatmul, NaiveMatmulConfig, naive_matmul},
        },
        global::memory::{GlobalLayout, GlobalLayoutConfig, GlobalLayoutLaunch, GlobalScaleLayout},
    },
    definition::{
        MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSelection,
        MatmulSetupError, MatrixLayout,
    },
    launch::{InputRuntimeArg, MatmulArgs, MatmulInputHandleRef, OutputRuntimeArg},
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}

impl BatchMatmulFamily for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulPrecision> = NaiveMatmul;
    type Config = NaiveMatmulConfig;
    type Blueprint = ();

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
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
        todo!()
        // unsafe {
        //     naive_matmul::launch_unchecked(
        //         client,
        //         cube_count,
        //         cube_dim,
        //         lhs_view,
        //         rhs_view,
        //         out.as_tensor_arg(1),
        //         *dtypes.lhs_global,
        //         *dtypes.acc_register,
        //         *dtypes.acc_global,
        //     )
        // }
    }
}
// naive_matmul::launch_unchecked(
//             client,
//             cube_count,
//             CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
//             lhs_view,
//             rhs_view,
//             out.as_tensor_arg(1),
//             *dtypes.lhs_global,
//             *dtypes.acc_register,
//             *dtypes.acc_global,
//         )
