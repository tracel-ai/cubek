//! The reduce operation, [`Reduce`]: built from a [`ReduceOpStrategy`], it launches the kernel the
//! strategy picks on the tile routines.

use crate::ReduceError;
use crate::kernels::{
    ReduceElements, ReduceKernelHeuristic, ReduceOpStrategy, ReduceTensors, launch_tuned, select,
};
use cubecl::Device;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;

/// A reduction of one tensor along one axis, run by the kernel its strategy picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduce {
    strategy: ReduceOpStrategy,
}

impl Reduce {
    /// The reduction whose kernel `strategy` picks.
    pub fn new(strategy: ReduceOpStrategy) -> Self {
        Self { strategy }
    }

    /// Reduce dim `axis` of `input` into `output` by `operation`, in `elements`, on `device`.
    /// `output` has `input`'s shape with `axis` one wide, `k` wide for a top-k.
    ///
    /// # Errors
    ///
    /// Where `axis` or `output`'s shape does not fit `input`, and [`ReduceError::Validation`]
    /// with the kernel's reason where a [`Specific`](ReduceOpStrategy::Specific) kernel declines
    /// the problem.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn execute(
        &self,
        client: &Client,
        device: &Device,
        input: TensorBinding,
        output: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Result<(), ReduceError> {
        validate_axis(input.shape.len(), axis)?;
        validate_shape(
            &input.shape,
            &output.shape,
            axis,
            operation.written_per_row(),
        )?;
        let tensors = ReduceTensors::new(client, input, output, axis, operation, elements);
        launched(self.strategy, device, tensors)
    }

    /// [`execute`](Self::execute), the values written to `values` in `elements.out` and their
    /// indices to `indices` in `index_element`, from one launch. `indices` has `values`' shape and
    /// strides. `Max`, `Min` and `TopK` write the same pair as `ArgMax`, `ArgMin` and `ArgTopK`.
    ///
    /// # Errors
    ///
    /// As [`execute`](Self::execute), and [`ReduceError::IndicesUnsupported`] for an operation
    /// whose values carry no index.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn execute_with_indices(
        &self,
        client: &Client,
        device: &Device,
        input: TensorBinding,
        values: TensorBinding,
        indices: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        index_element: ElemType,
    ) -> Result<(), ReduceError> {
        let operation = indexed(operation)?;
        validate_axis(input.shape.len(), axis)?;
        validate_shape(
            &input.shape,
            &values.shape,
            axis,
            operation.written_per_row(),
        )?;
        validate_indices(&values, &indices)?;
        let tensors = ReduceTensors::with_indices(
            client,
            input,
            values,
            indices,
            axis,
            operation,
            elements,
            index_element,
        );
        launched(self.strategy, device, tensors)
    }
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

/// The operation that writes `operation`'s values with their indices: the value and its index
/// accumulate into the same carrier, so `Max` is `ArgMax` with both halves written.
///
/// # Errors
///
/// [`ReduceError::IndicesUnsupported`] for an operation whose values carry no index.
fn indexed(operation: ReduceOperation) -> Result<ReduceOperation, ReduceError> {
    match operation {
        ReduceOperation::Max | ReduceOperation::ArgMax => Ok(ReduceOperation::ArgMax),
        ReduceOperation::Min | ReduceOperation::ArgMin => Ok(ReduceOperation::ArgMin),
        ReduceOperation::TopK { k } | ReduceOperation::ArgTopK { k } => {
            Ok(ReduceOperation::ArgTopK { k })
        }
        ReduceOperation::Sum => Err(unindexed("Sum")),
        ReduceOperation::Prod => Err(unindexed("Prod")),
        ReduceOperation::Mean => Err(unindexed("Mean")),
        ReduceOperation::MaxAbs => Err(unindexed("MaxAbs")),
        ReduceOperation::Any => Err(unindexed("Any")),
        ReduceOperation::All => Err(unindexed("All")),
    }
}

/// Why [`Reduce::execute_with_indices`] refuses the operation `operation_name`.
fn unindexed(operation_name: &'static str) -> ReduceError {
    ReduceError::IndicesUnsupported {
        operation: operation_name,
    }
}

/// Whether `axis` is a dim of a tensor of rank `rank`.
fn validate_axis(rank: usize, axis: usize) -> Result<(), ReduceError> {
    if axis >= rank {
        return Err(ReduceError::InvalidAxis { axis, rank });
    }
    Ok(())
}

/// Whether `output_shape` is `input_shape` with `axis` `written_per_row` wide, and the reduced dim
/// at least that long.
fn validate_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    written_per_row: usize,
) -> Result<(), ReduceError> {
    if input_shape[axis] < written_per_row {
        return Err(ReduceError::ReduceAxisTooSmall {
            axis_length: input_shape[axis],
            k: written_per_row,
        });
    }
    let mut expected_shape = input_shape.to_vec();
    expected_shape[axis] = written_per_row;
    if output_shape != expected_shape {
        return Err(ReduceError::MismatchOutputShape {
            expected_shape,
            output_shape: output_shape.to_vec(),
        });
    }
    Ok(())
}

/// Whether `indices` has `values`' shape and strides: both outputs are written through one
/// layout, derived from the values alone.
fn validate_indices(values: &TensorBinding, indices: &TensorBinding) -> Result<(), ReduceError> {
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
    Ok(())
}
