use std::marker::PhantomData;

use crate::components::batch::BatchMatmulFamily;
use crate::components::batch::partitioned_matmul::config::PartitionedBatchConfig;
use crate::components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::partitioned_matmul::matmul::matmul_entry;
use crate::components::batch::partitioned_matmul::partition::GlobalPartitionMatmul;
use crate::components::global::GlobalMatmulFamily;
use crate::definition::CubeCountInputArgs;
use crate::definition::TilingBlueprint;
use crate::definition::{
    MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSetupError,
};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use cubecl::prelude::*;

/// Simple partitioned batch matmul family for any precision
pub struct PartitionedBatchMatmulFamily<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul> {
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

impl<GMM: GlobalMatmulFamily, S: GlobalPartitionMatmul> BatchMatmulFamily
    for PartitionedBatchMatmulFamily<GMM, S>
{
    type Matmul<MP: MatmulPrecision> = PartitionedBatchMatmul<MP, GMM::Matmul<MP>, S>;
    type Config = PartitionedBatchConfig<GMM::Config>;
    type Blueprint = TilingBlueprint;

    fn expand_config<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        blueprint: &Self::Blueprint,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::expand_config(client, problem, blueprint, line_sizes, dtypes)?;

        PartitionedBatchConfig::new(
            global_config,
            blueprint
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
            blueprint.tiling_scheme.global_partition_size,
        )
        .validate(problem)
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
            matmul_entry::launch_unchecked::<MA, GMM, S, R>(
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
