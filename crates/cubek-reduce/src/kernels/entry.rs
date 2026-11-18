//! The reduce operation: the kernel a [`ReduceOpStrategy`] picks, launched on the tile routines.

use crate::kernels::{
    ReduceElements, ReduceKernelHeuristic, ReduceOpStrategy, ReduceTensors, launch_tuned, select,
};
use crate::{ReduceDtypes, ReduceError, ReduceOperationConfig, ReduceWithIndicesDtypes};
use cubecl::Device;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;

/// Reduce dim `axis` of `input` into `output` by `operation`, the kernel picked as `strategy`
/// says on `device`. `output` has `input`'s shape with `axis` one wide, `k` wide for a top-k.
///
/// # Errors
///
/// Where `axis` or `output`'s shape does not fit `input`, and [`ReduceError::Validation`] with
/// the kernel's reason where a [`Specific`](ReduceOpStrategy::Specific) kernel declines the
/// problem.
#[allow(clippy::too_many_arguments)]
pub fn reduce(
    client: &Client,
    device: &Device,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceOpStrategy,
    operation: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) -> Result<(), ReduceError> {
    crate::validate_axis(input.shape.len(), axis)?;
    crate::validate_shapes(
        &input.shape,
        &output.shape,
        axis,
        match operation {
            ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => Some(k),
            _ => None,
        },
    )?;
    let tensors = ReduceTensors::new(
        client,
        input,
        output,
        axis,
        operation_of(operation),
        ReduceElements {
            input: dtypes.input,
            accumulator: dtypes.accumulation,
            out: dtypes.output,
        },
    );
    launched(strategy, device, tensors)
}

/// [`reduce`], the values written to `values` and their indices to `indices` from one launch.
/// `indices` has `values`' shape and strides.
///
/// # Errors
///
/// As [`reduce`], and [`ReduceError::IndicesUnsupported`] for an operation whose values carry no
/// index.
#[allow(clippy::too_many_arguments)]
pub fn reduce_with_indices(
    client: &Client,
    device: &Device,
    input: TensorBinding,
    values: TensorBinding,
    indices: TensorBinding,
    axis: usize,
    strategy: ReduceOpStrategy,
    operation: ReduceOperationConfig,
    dtypes: ReduceWithIndicesDtypes,
) -> Result<(), ReduceError> {
    let k = match operation {
        ReduceOperationConfig::TopK(k) | ReduceOperationConfig::ArgTopK(k) => k,
        ReduceOperationConfig::Max
        | ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::Min
        | ReduceOperationConfig::ArgMin => 1,
        other => {
            return Err(ReduceError::IndicesUnsupported {
                operation: crate::operation_name(&other),
            });
        }
    };
    crate::validate_axis(input.shape.len(), axis)?;
    crate::validate_shapes(&input.shape, &values.shape, axis, Some(k))?;
    if indices.shape.as_slice() != values.shape.as_slice() {
        return Err(ReduceError::MismatchIndicesShape {
            values_shape: values.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
        });
    }
    if indices.strides != values.strides {
        return Err(ReduceError::MismatchIndicesStrides {
            values_strides: values.strides.to_vec(),
            indices_strides: indices.strides.to_vec(),
        });
    }
    // Max and ArgMax accumulate into the same carrier; with both halves written they are one launch.
    let operation = match operation_of(operation) {
        ReduceOperation::Max => ReduceOperation::ArgMax,
        ReduceOperation::Min => ReduceOperation::ArgMin,
        ReduceOperation::TopK { k } => ReduceOperation::ArgTopK { k },
        other => other,
    };
    let tensors = ReduceTensors::with_indices(
        client,
        input,
        values,
        indices,
        axis,
        operation,
        ReduceElements {
            input: dtypes.input,
            accumulator: dtypes.accumulation,
            out: dtypes.values,
        },
        dtypes.indices,
    );
    launched(strategy, device, tensors)
}

/// `tensors` reduced as `strategy` picks the kernel, on `device`.
fn launched(
    strategy: ReduceOpStrategy,
    device: &Device,
    tensors: ReduceTensors,
) -> Result<(), ReduceError> {
    let hw = Hardware::from_client(tensors.client());
    let launched = match strategy {
        ReduceOpStrategy::Autotune => {
            launch_tuned(tensors, device);
            Ok(())
        }
        ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile) => select(&hw, &tensors),
        ReduceOpStrategy::Specific(kernel) => kernel.launch(&hw, &tensors),
    };
    launched.map_err(|why| ReduceError::Validation {
        details: match why {
            Refusal::Declined { why } => why,
            _ => "the reduce kernel cannot launch this problem on this device",
        },
    })
}

/// Today's operation, as the microkernels name it.
fn operation_of(config: ReduceOperationConfig) -> ReduceOperation {
    match config {
        ReduceOperationConfig::Sum => ReduceOperation::Sum,
        ReduceOperationConfig::Prod => ReduceOperation::Prod,
        ReduceOperationConfig::Mean => ReduceOperation::Mean,
        ReduceOperationConfig::MaxAbs => ReduceOperation::MaxAbs,
        ReduceOperationConfig::ArgMax => ReduceOperation::ArgMax,
        ReduceOperationConfig::ArgMin => ReduceOperation::ArgMin,
        ReduceOperationConfig::Max => ReduceOperation::Max,
        ReduceOperationConfig::Min => ReduceOperation::Min,
        ReduceOperationConfig::ArgTopK(k) => ReduceOperation::ArgTopK { k },
        ReduceOperationConfig::TopK(k) => ReduceOperation::TopK { k },
        ReduceOperationConfig::Any => ReduceOperation::Any,
        ReduceOperationConfig::All => ReduceOperation::All,
    }
}
