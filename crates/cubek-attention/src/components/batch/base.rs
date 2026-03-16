use cubecl;
use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;
use cubecl::std::tensor::r#virtual::VirtualTensor;

use crate::definition::{
    AttentionElems, AttentionPrecision, AttentionSetupError, CubeCountInput, InputRuntimeArg,
    OutputRuntimeArg,
};
use crate::definition::{CubeCountInputArgs, attention_types::*};
use crate::launch::AttentionArgs;
use crate::{components::global::GlobalAttentionConfig, definition::AttentionVectorSizes};
use std::{fmt::Debug, hash::Hash};

/// A family of [BatchAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait BatchAttentionFamily: Send + Sync + 'static {
    /// The specific [BatchAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: BatchAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: BatchAttentionConfig;
    type Blueprint;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<AA: AttentionArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<AA, R>,
        output: OutputRuntimeArg<AA, R>,
        cube_count_input: CubeCountInputArgs<R>,
        dtypes: &AttentionElems,
        vector_sizes: &AttentionVectorSizes,
        attention_blueprint: Self::Blueprint,
    ) -> Result<(), LaunchError>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: Self::Blueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;
}

#[cube]
pub trait BatchAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Attention.
    type Config: BatchAttentionConfig;

    fn execute(
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Batch Attention level
pub trait BatchAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type GlobalConfig: GlobalAttentionConfig;

    fn global_config(&self) -> Self::GlobalConfig;

    fn cube_dim(&self) -> CubeDim;
}
