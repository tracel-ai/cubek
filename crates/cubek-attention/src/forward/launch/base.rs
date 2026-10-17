use cubecl::{client::Client, prelude::TensorBinding};

use crate::forward::routines::{
    Routine, blackbox_accelerated::BlackboxAcceleratedRoutine, unit::UnitRoutine,
};
use crate::{
    forward::definition::AttentionSetupError,
    forward::definition::{
        AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
    },
    forward::launch::args::{TensorArgs, TensorInputsLaunch},
    forward::routines::DeviceSettings,
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
pub fn launch_ref(
    strategy: Strategy,
    client: &Client,
    query: TensorBinding,
    key: TensorBinding,
    value: TensorBinding,
    mask: Option<TensorBinding>,
    out: TensorBinding,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError> {
    match strategy {
        Strategy::BlackboxAccelerated(strategy) => {
            let unmasked_compatible = mask.is_none()
                && query.shape.len() == 4
                && key.shape.len() == 4
                && value.shape.len() == 4
                && query.shape[3] == key.shape[3]
                && query.shape[3] == value.shape[3];
            let fallback_inputs = unmasked_compatible.then(|| {
                (
                    query.clone(),
                    key.clone(),
                    value.clone(),
                    out.clone(),
                    attention_options.causal,
                )
            });
            match launch_attention::<BlackboxAcceleratedRoutine>(
                client,
                query,
                key,
                value,
                mask,
                out,
                attention_global_types,
                strategy,
                attention_options,
            ) {
                Ok(()) => Ok(()),
                Err(AttentionSetupError::InvalidConfig(_)) if fallback_inputs.is_some() => {
                    let (q, k, v, o, causal) = fallback_inputs.unwrap();
                    let lse_shape = vec![q.shape[0], q.shape[1], q.shape[2]];
                    let lse_elems: usize = lse_shape.iter().product();
                    let f32_ty = <f32 as cubecl::prelude::Scalar>::elem_type_native();
                    let lse_handle = cubecl::std::tensor::TensorHandle::new_contiguous(
                        lse_shape,
                        client.empty(lse_elems * f32_ty.size()),
                        f32_ty,
                    );
                    let scale = (q.shape[3] as f32).sqrt().recip();
                    super::lse::flash_attention_forward_lse(
                        client,
                        q,
                        k,
                        v,
                        o,
                        lse_handle.binding(),
                        attention_global_types,
                        causal,
                        scale,
                    )
                }
                Err(err) => Err(err),
            }
        }
        Strategy::Unit(strategy) => launch_attention::<UnitRoutine>(
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
pub fn launch_attention<A: Routine>(
    client: &Client,
    query: TensorBinding,
    key: TensorBinding,
    value: TensorBinding,
    mask: Option<TensorBinding>,
    out: TensorBinding,
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
        <A as Routine>::BatchAttention::launch_unchecked::<TensorArgs>(
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
            cubek_std::cube_count::cube_mapping_launch(&launch_info.cube_count_plan),
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
