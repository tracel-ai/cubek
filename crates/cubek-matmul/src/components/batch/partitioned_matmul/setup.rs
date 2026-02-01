use std::marker::PhantomData;

use crate::components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul;
use crate::components::batch::partitioned_matmul::matmul::matmul_entry;
use crate::components::batch::partitioned_matmul::partition::GlobalPartitionMatmul;
use crate::components::global::GlobalMatmulFamily;
use crate::components::{
    batch::partitioned_matmul::config::PartitionedBatchConfig, stage::NumStages,
};
use crate::definition::CubeMappingLaunch;
use crate::definition::MatmulLineSizes;
use crate::definition::MatmulProblem;
use crate::definition::TilingBlueprint;
use crate::definition::{MatmulElems, MatmulPrecision, MatmulSetupError};
use crate::launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use crate::{components::batch::BatchMatmulFamily, launch::ConfigRuntimeArg};
use cubecl::{ir::DeviceProperties, prelude::*};

/// Simple partitioned batch matmul family for any precision
pub struct PartitionedBatchMatmulFamily<
    RC: RuntimeConfig,
    GMM: GlobalMatmulFamily<RC>,
    S: GlobalPartitionMatmul,
> {
    _rc: PhantomData<RC>,
    _gmm: PhantomData<GMM>,
    _s: PhantomData<S>,
}

impl<RC: RuntimeConfig, GMM: GlobalMatmulFamily<RC>, S: GlobalPartitionMatmul> BatchMatmulFamily<RC>
    for PartitionedBatchMatmulFamily<RC, GMM, S>
{
    type Matmul<MP: MatmulPrecision> = PartitionedBatchMatmul<RC, MP, GMM::Matmul<MP>, S>;
    type Config = PartitionedBatchConfig<GMM::Config>;
    type Blueprint = TilingBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::expand_config(device_props, blueprint, dtypes, line_sizes)?;

        Ok(PartitionedBatchConfig::new(
            global_config,
            blueprint.tiling_scheme.global_partition_size,
        ))
    }

    fn num_stages() -> NumStages {
        GMM::num_stages()
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        config: ConfigRuntimeArg<'a, MA, R>,
        cube_count_input: CubeMappingLaunch<'a, R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, GMM, S, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                config,
                cube_count_input,
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
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        GMM::cubedim_resource(blueprint, dtypes, line_sizes)
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError> {
        GMM::validate_blueprint(client, blueprint, problem, dtypes, line_sizes)
    }
}
