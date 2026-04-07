use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::routines::{
    Routine, blackbox_accelerated::BlackboxAcceleratedRoutine, unit::UnitRoutine,
};
use crate::{
    definition::AttentionSetupError,
    definition::{AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem},
    launch::args::{TensorArgs, TensorInputsLaunch},
    routines::DeviceSettings,
};

use crate::components::batch::BatchAttentionFamily;

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    /// Use a predefined blueprint
    Forced(R::Blueprint),
    /// Allows to give limited settings information, and the rest is inferred from it
    Inferred(R::Strategy),
}

#[derive(Debug, Clone)]
pub enum Strategy {
    BlackboxAccelerated(BlueprintStrategy<BlackboxAcceleratedRoutine>),
    Unit(BlueprintStrategy<UnitRoutine>),
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: TensorBinding<R>,
    key: TensorBinding<R>,
    value: TensorBinding<R>,
    mask: Option<TensorBinding<R>>,
    out: TensorBinding<R>,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::BlackboxAccelerated(strategy) => {
            launch_attention::<R, BlackboxAcceleratedRoutine>(
                client,
                query,
                key,
                value,
                mask,
                out,
                attention_global_types,
                strategy,
                attention_options,
            )
        }
        Strategy::Unit(strategy) => launch_attention::<R, UnitRoutine>(
            client,
            query,
            key,
            value,
            mask,
            out,
            attention_global_types,
            strategy,
            attention_options,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_attention<R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    query: TensorBinding<R>,
    key: TensorBinding<R>,
    value: TensorBinding<R>,
    mask: Option<TensorBinding<R>>,
    out: TensorBinding<R>,
    global_dtypes: &AttentionGlobalTypes,
    strategy: BlueprintStrategy<A>,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    let definition = AttentionProblem {
        dims: AttentionDims {
            batch: query.shape[0],
            num_heads: query.shape[1],
            seq_q: query.shape[2],
            head_dim: query.shape[3],
            seq_kv: key.shape[2],
            val_dim: value.shape[3],
        },
        masked: mask.is_some(),
        global_dtypes: global_dtypes.clone(),
        options: attention_options,
        address_type: [
            query.required_address_type(global_dtypes.query.size()),
            key.required_address_type(global_dtypes.key.size()),
            value.required_address_type(global_dtypes.value.size()),
            mask.as_ref()
                .map(|mask| mask.required_address_type(global_dtypes.mask.size()))
                .unwrap_or_default(),
            out.required_address_type(global_dtypes.out.size()),
        ]
        .into_iter()
        .max()
        .unwrap_or_default(),
    };

    let device_settings = DeviceSettings::new(client, &definition);

    let launch_info = A::prepare(&definition, &device_settings, strategy)?;

    // This allows an expand_config error to be caught by the client rather than the server.
    // Then the server can re-run expand config assuming a valid blueprint
    <A as Routine>::BatchAttention::expand_config(
        client.properties(),
        launch_info.blueprint.clone(),
        &launch_info.dtypes,
    )?;

    let result = unsafe {
        <A as Routine>::BatchAttention::launch_unchecked::<TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            launch_info.address_type,
            TensorInputsLaunch::new(
                query.into_tensor_arg(),
                key.into_tensor_arg(),
                value.into_tensor_arg(),
                mask.map(|it| it.into_tensor_arg()).into(),
            ),
            out.into_tensor_arg(),
            launch_info.cube_count_plan.as_args(),
            &launch_info.dtypes,
            &device_settings.vector_sizes,
            launch_info.blueprint,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(AttentionSetupError::Execution(err)),
    }
}
