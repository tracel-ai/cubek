use crate::{
    definitions::{
        PrefixScanBlueprint, PrefixScanKind, PrefixScanLaunchSettings, PrefixScanSettings,
    },
    error::PrefixScanError,
    eval::strategy::{GlobalPrefixScanStrategy, PrefixScanStrategy},
    instructions::{
        Maximum, Minimum, PrefixPrefixScanInstructionKind, PrefixScanInstruction, Product, Sum,
    },
    kernels::single_pass_prefix_scan,
};
use cubecl::{client::Client, prelude::*, std::tensor::TensorHandle};

pub fn prefix_scan_launch_generic_instruction<I: PrefixScanInstruction>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: PrefixScanKind,
    strategy: PrefixScanStrategy,
) -> Result<(), PrefixScanError> {
    if input.shape.rank() != 1 {
        return Err(PrefixScanError::UnsupportedRank {
            rank: input.shape.rank(),
        });
    }
    if input.shape != output.shape {
        return Err(PrefixScanError::InputOutputShapeMismatch {
            input: input.shape.to_vec(),
            output: output.shape.to_vec(),
        });
    }
    let settings =
        PrefixScanSettings::new(client.properties(), dtype, &input.shape, &input.strides);
    let launch_settings = PrefixScanLaunchSettings::new(client, &input.shape, &settings)?;
    let reductions = match strategy.global {
        GlobalPrefixScanStrategy::StreamScan => None,
        GlobalPrefixScanStrategy::DecoupledLookback => Some(TensorHandle::empty(
            client,
            [launch_settings.working_cubes],
            dtype,
        )),
    };
    let task_allocator = TensorHandle::zeros(client, [1], u32::elem_type_native());
    let task_status = TensorHandle::zeros(
        client,
        [launch_settings.working_cubes],
        u32::elem_type_native(),
    );
    unsafe {
        single_pass_prefix_scan::launch_unchecked::<I>(
            client,
            launch_settings.cube_count,
            launch_settings.cube_dim,
            settings.vector_size as usize,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            reductions
                .map(|tensor_handle| tensor_handle.into_arg())
                .unwrap_or_else(|| output.clone().into_tensor_arg()),
            task_allocator.into_arg(),
            task_status.into_arg(),
            PrefixScanBlueprint::new(client.properties(), scan_kind, &settings, strategy)?,
            dtype,
        );
    }
    Ok(())
}

pub fn prefix_scan_launch(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: PrefixScanKind,
    scan_instruction_kind: PrefixPrefixScanInstructionKind,
    strategy: PrefixScanStrategy,
) -> Result<(), PrefixScanError> {
    match scan_instruction_kind {
        PrefixPrefixScanInstructionKind::Sum => prefix_scan_launch_generic_instruction::<Sum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        PrefixPrefixScanInstructionKind::Product => {
            prefix_scan_launch_generic_instruction::<Product>(
                client, input, output, dtype, scan_kind, strategy,
            )
        }
        PrefixPrefixScanInstructionKind::Minimum => {
            prefix_scan_launch_generic_instruction::<Minimum>(
                client, input, output, dtype, scan_kind, strategy,
            )
        }
        PrefixPrefixScanInstructionKind::Maximum => {
            prefix_scan_launch_generic_instruction::<Maximum>(
                client, input, output, dtype, scan_kind, strategy,
            )
        }
    }
}
