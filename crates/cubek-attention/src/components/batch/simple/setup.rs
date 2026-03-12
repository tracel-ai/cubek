use std::marker::PhantomData;

use cubecl::{
    ir::{AddressType, DeviceProperties},
    server::LaunchError,
};

use crate::{
    components::{
        batch::{
            BatchAttentionFamily,
            entry_point::attention,
            simple::{SimpleBatchAttention, config::SimpleBatchConfig},
        },
        global::GlobalAttentionFamily,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        AttentionVectorSizes, CubeCountInputArgs, InputRuntimeArg, OutputRuntimeArg,
    },
    launch::AttentionArgs,
};

pub struct SimpleBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for SimpleBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = SimpleBatchAttention<AP, GA::Attention<AP>>;
    type Config = SimpleBatchConfig<GA::Config>;
    type Blueprint = AttentionBlueprint;

    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: cubecl::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        cube_dim: cubecl::CubeDim,
        cube_count: cubecl::CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<AA, R>,
        output: OutputRuntimeArg<AA, R>,
        cube_count_input: CubeCountInputArgs<R>,
        dtypes: &AttentionElems,
        vector_sizes: &AttentionVectorSizes,
        blueprint: AttentionBlueprint,
    ) -> Result<(), LaunchError> {
        unsafe {
            attention::launch_unchecked::<AA, Self, R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                cube_count_input,
                blueprint,
                dtypes.clone(),
                dtypes.into(),
                vector_sizes.into(),
            )
        };

        Ok(())
    }

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let global_config = GA::expand_config(device_props, &blueprint, dtypes)?;

        Ok(SimpleBatchConfig::new(global_config))
    }
}
