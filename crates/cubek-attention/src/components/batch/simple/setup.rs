use std::marker::PhantomData;

use cubecl::server::LaunchError;

use crate::{
    components::{
        batch::{
            BatchAttentionFamily,
            entry_point::attention,
            simple::{
                SimpleBatchAttention,
                config::{HypercubeConfig, SimpleBatchConfig},
            },
        },
        global::GlobalAttentionFamily,
    },
    launch::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        CubeCountInputArgs, InputRuntimeArg, OutputRuntimeArg, args::AttentionArgs,
    },
};

pub struct SimpleBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for SimpleBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = SimpleBatchAttention<AP, GA::Attention<AP>>;
    type Config = SimpleBatchConfig<GA::Config>;

    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: cubecl::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        cube_dim: cubecl::CubeDim,
        cube_count: cubecl::CubeCount,
        input: InputRuntimeArg<'a, AA, R>,
        output: OutputRuntimeArg<'a, AA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        dtypes: &AttentionElems,
        blueprint: AttentionBlueprint,
    ) -> Result<(), LaunchError> {
        unsafe {
            attention::launch_unchecked::<AA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                blueprint,
                dtypes.into(),
            )
        }
    }

    fn expand_blueprint(
        blueprint: AttentionBlueprint,
    ) -> Result<Self::Config, AttentionSetupError> {
        let global_config = GA::expand_blueprint(&blueprint)?;

        Ok(SimpleBatchConfig::new(
            global_config,
            HypercubeConfig::from_blueprint(blueprint.hypercube_blueprint),
        ))
    }
}
