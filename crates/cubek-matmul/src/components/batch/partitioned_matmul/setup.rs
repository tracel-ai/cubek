use std::marker::PhantomData;

use crate::components::{
    batch::partitioned_matmul::config::PartitionedBatchConfig, stage::NumStages,
};
use crate::{
    components::batch::partitioned_matmul::matmul::PartitionedBatchMatmul,
    components::batch::partitioned_matmul::matmul::matmul_entry,
    components::batch::partitioned_matmul::partition::GlobalPartitionMatmul,
    components::global::GlobalMatmulFamily,
};
use crate::{
    definition::CubeMappingLaunch,
    definition::MatmulProblem,
    definition::MatmulVectorSizes,
    definition::TilingBlueprint,
    definition::{MatmulElems, MatmulSetupError, MatmulTypes},
    launch::*,
    {components::CubeDimResource, launch::RuntimeConfig},
    {components::batch::BatchMatmulFamily, launch::ConfigRuntimeArg},
};
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
    type Matmul<MP: MatmulTypes> = PartitionedBatchMatmul<RC, MP, GMM::Matmul<MP>, S>;
    type Config = PartitionedBatchConfig<GMM::Config>;
    type Blueprint = TilingBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let global_config = GMM::expand_config(device_props, blueprint, dtypes, vector_sizes)?;

        Ok(PartitionedBatchConfig::new(
            global_config,
            blueprint.tiling_scheme.global_partition_size,
        ))
    }

    fn num_stages() -> NumStages {
        GMM::num_stages()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        config: ConfigRuntimeArg<MA, R>,
        cube_count_input: CubeMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, GMM, S, R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                config,
                cube_count_input,
                blueprint,
                dtypes.clone(),
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        GMM::cubedim_resource(blueprint, dtypes, vector_sizes)
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        GMM::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)
    }
}
