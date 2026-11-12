use crate::{
    base::{
        PrefixScanGlobalStrategy, PrefixScanKind, PrefixScanLaunchSettings, PrefixScanSettings,
    },
    blueprint::PrefixScanBlueprint,
    error::PrefixScanError,
    instructions::{
        Maximum, Minimum, PrefixScanInstruction, PrefixScanInstructionKind, Product, Sum,
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
    settings: &PrefixScanSettings,
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
    let launch_settings = PrefixScanLaunchSettings::new(client, &input.shape, settings)?;
    let reductions = match settings.strategy.global {
        PrefixScanGlobalStrategy::StreamScan => None,
        PrefixScanGlobalStrategy::DecoupledLookback => Some(TensorHandle::empty(
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
            launch_settings.cube_count.clone(),
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
            PrefixScanBlueprint::new(client.properties(), scan_kind, settings)?,
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
    scan_instruction_kind: PrefixScanInstructionKind,
    settings: &PrefixScanSettings,
) -> Result<(), PrefixScanError> {
    match scan_instruction_kind {
        PrefixScanInstructionKind::Sum => prefix_scan_launch_generic_instruction::<Sum>(
            client, input, output, dtype, scan_kind, settings,
        ),
        PrefixScanInstructionKind::Product => prefix_scan_launch_generic_instruction::<Product>(
            client, input, output, dtype, scan_kind, settings,
        ),
        PrefixScanInstructionKind::Minimum => prefix_scan_launch_generic_instruction::<Minimum>(
            client, input, output, dtype, scan_kind, settings,
        ),
        PrefixScanInstructionKind::Maximum => prefix_scan_launch_generic_instruction::<Maximum>(
            client, input, output, dtype, scan_kind, settings,
        ),
    }
}
