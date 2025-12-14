use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use cubecl::std::tensor::TensorHandle;

use crate::launch::args::{TensorArgs, TensorInputsLaunch};
use crate::launch::definition::{
    AttentionDefinition, AttentionDims, AttentionGlobalTypes, AttentionOptions,
};
use crate::launch::{AttentionBlueprint, AttentionSetupError};
use crate::routines::DeviceSettings;
use crate::routines::{
    Routine, blackbox_accelerated::BlackboxAcceleratedRoutine, unit::UnitRoutine,
};

use crate::components::batch::BatchAttentionFamily;

#[derive(Debug, Clone)]
pub enum RoutineStrategy<R: Routine> {
    /// Use a predefined blueprint
    Forced(AttentionBlueprint),
    /// Allows to give limited settings information, and the rest is inferred from it
    Inferred(R::Strategy),
}

#[derive(Debug, Clone)]
pub enum Strategy {
    BlackboxAccelerated(RoutineStrategy<BlackboxAcceleratedRoutine>),
    Unit(RoutineStrategy<UnitRoutine>),
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: TensorHandle<R>,
    key: TensorHandle<R>,
    value: TensorHandle<R>,
    mask: Option<TensorHandle<R>>,
    out: TensorHandle<R>,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    launch_ref(
        strategy,
        client,
        &query.as_ref(),
        &key.as_ref(),
        &value.as_ref(),
        &mask.as_ref().map(|m| m.as_ref()),
        &out.as_ref(),
        attention_global_types,
        attention_options,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
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
    query: &TensorHandleRef<R>,
    key: &TensorHandleRef<R>,
    value: &TensorHandleRef<R>,
    mask: &Option<TensorHandleRef<R>>,
    out: &TensorHandleRef<R>,
    global_dtypes: &AttentionGlobalTypes,
    strategy: RoutineStrategy<A>,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    let definition = AttentionDefinition {
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
    };

    let device_settings = DeviceSettings::new(client, &definition);
    let launch_info = A::prepare(&definition, &device_settings, strategy)?;

    let result = unsafe {
        <A as Routine>::BatchAttention::launch_unchecked::<TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            TensorInputsLaunch::new(
                query.as_tensor_arg(device_settings.line_sizes.query),
                key.as_tensor_arg(device_settings.line_sizes.key),
                value.as_tensor_arg(device_settings.line_sizes.value),
                mask.as_ref()
                    .map(|it| it.as_tensor_arg(device_settings.line_sizes.mask))
                    .into(),
            ),
            out.as_tensor_arg(device_settings.line_sizes.out),
            launch_info.cube_count_plan.as_args(),
            &launch_info.dtypes,
            launch_info.blueprint,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(AttentionSetupError::Execution(err)),
    }
}
