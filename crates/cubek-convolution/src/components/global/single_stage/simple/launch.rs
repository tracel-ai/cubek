use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, server::LaunchError};
use cubek_matmul::launch::{InputRuntimeArg, OutputRuntimeArg};
use cubek_matmul::{
    components::{
        global::PartitionedStageFamily,
        stage::{StageMatmulFamily, StridedStageFamily},
    },
    definition::MatmulElems,
    launch::MatmulArgs,
};

use crate::components::global::{
    GlobalConfig,
    args::RuntimeArgsLaunch,
    entry_point::{ConvolutionLaunch, implicit_conv},
    read::full_reader::FullLoadingStrategy,
    single_stage::simple::SimpleConvolutionFamily,
};

impl<
    SMM: StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
    LL: FullLoadingStrategy,
    LR: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
> ConvolutionLaunch<GlobalConfig<Self>> for SimpleConvolutionFamily<SMM, LL, LR>
{
    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        runtime_args: RuntimeArgsLaunch<'a, R>,
        config: GlobalConfig<Self>,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        unsafe {
            implicit_conv::launch_unchecked::<MA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                runtime_args,
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
