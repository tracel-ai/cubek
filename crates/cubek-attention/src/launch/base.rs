use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};

use crate::definition::AttentionSetupError;
use crate::definition::{AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem};
use crate::launch::args::{TensorArgs, TensorInputsLaunch};
use crate::routines::DeviceSettings;
use crate::routines::whitebox_accelerated::WhiteboxAcceleratedRoutine;
use crate::routines::{
    Routine, blackbox_accelerated::BlackboxAcceleratedRoutine, unit::UnitRoutine,
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
    WhiteboxAccelerated(BlueprintStrategy<WhiteboxAcceleratedRoutine>),
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
        Strategy::WhiteboxAccelerated(strategy) => {
            launch_attention::<R, WhiteboxAcceleratedRoutine>(
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
            query.required_address_type(),
            key.required_address_type(),
            value.required_address_type(),
            mask.as_ref()
                .map(|mask| mask.required_address_type())
                .unwrap_or_default(),
            out.required_address_type(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default(),
    };

    let device_settings = DeviceSettings::new(client, &definition);
    let launch_info = A::prepare(&definition, &device_settings, strategy)?;

    let result = unsafe {
        <A as Routine>::BatchAttention::launch_unchecked::<TensorArgs, R>(
            client,
            launch_info.cube_dim,
            launch_info.cube_count_plan.resolve(),
            launch_info.address_type,
            TensorInputsLaunch::new(
                query.into_tensor_arg(device_settings.line_sizes.query),
                key.into_tensor_arg(device_settings.line_sizes.key),
                value.into_tensor_arg(device_settings.line_sizes.value),
                mask.map(|it| it.into_tensor_arg(device_settings.line_sizes.mask))
                    .into(),
            ),
            out.into_tensor_arg(device_settings.line_sizes.out),
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
