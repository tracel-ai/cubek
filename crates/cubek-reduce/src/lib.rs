//! This provides different implementations of the reduce algorithm which
//! can run on multiple GPU backends using CubeCL.
//!
//! A reduction is a tensor operation mapping a rank `R` tensor to a rank `R - 1`
//! by agglomerating all elements along a given axis with some binary operator.
//! This is often also called folding.
//!
//! This crate provides a main entrypoint as the [`reduce`] function which allows to automatically
//! perform a reduction for a given instruction implementing the [`ReduceInstruction`] trait and a given [`ReduceStrategy`].
//! It also provides implementation of the [`ReduceInstruction`] trait for common operations in the [`instructions`] module.
//! Finally, it provides many reusable primitives to perform different general reduction algorithms in the [`primitives`] module.

#![allow(
    clippy::type_complexity,
    reason = "Too sensitive, triggers on tuple of vector."
)]

pub mod components;
pub mod launch;
pub mod routines;

mod error;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;

pub use crate::launch::{ReduceStrategy, ReduceWithIndicesDtypes};
use crate::{
    components::instructions::ReduceOperationConfig,
    launch::{launch_reduce, launch_reduce_with_indices},
};
pub use components::{
    args::init_tensors,
    config::*,
    instructions::{ReduceFamily, ReduceInstruction},
    precision::ReducePrecision,
};
use cubecl::prelude::*;
pub use error::*;
pub use launch::{ReduceDtypes, reduce_kernel};
pub use routines::shared_sum::shared_sum;

/// Reduce the given `axis` of the `input` tensor using the instruction `Inst` and write the result into `output`.
///
/// An optional [`ReduceStrategy`] can be provided to force the reduction to use a specific algorithm. If omitted, a best effort
/// is done to try and pick the best strategy supported for the provided `client`.
///
/// Return an error if `strategy` is `Some(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
/// The shape of `output` must be the same as input except with a value of 1 for the given `axis`.
///
///
/// # Example
///
/// This examples show how to sum the rows of a small `2 x 2` matrix into a `1 x 2` vector.
/// For more details, see the CubeCL documentation.
///
/// ```ignore
/// use cubecl_reduce::instructions::Sum;
///
/// let client = /* ... */;
/// let size_f32 = std::mem::size_of::<f32>();
/// let axis = 0; // 0 for rows, 1 for columns in the case of a matrix.
///
/// // Create input and output handles.
/// let input_handle = client.create(f32::as_bytes(&[0, 1, 2, 3]));
/// let input = unsafe {
///     TensorBinding::from_raw_parts(
///         &input_handle,
///         &[2, 1],
///         &[2, 2],
///         size_f32,
///     )
/// };
///
/// let output_handle = client.empty(2 * size_f32);
/// let output = unsafe {
///     TensorBinding::from_raw_parts(
///         &output_handle,
///         &output_stride,
///         &output_shape,
///         size_f32,
///     )
/// };
///
/// // Here `R` is a `cubecl::Runtime`.
/// let result = reduce::<R, f32, f32, Sum>(&client, input, output, axis, None);
///
/// if result.is_ok() {
///        let binding = output_handle.binding();
///        let bytes = client.read_one(binding);
///        let output_values = f32::from_bytes(&bytes);
///        println!("Output = {:?}", output_values); // Should print [1, 5].
/// }
/// ```
pub fn reduce<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
    operation: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) -> Result<(), ReduceError> {
    validate_axis(input.shape.len(), axis)?;
    validate_shapes(
        &input.shape,
        &output.shape,
        axis,
        match operation {
            ReduceOperationConfig::ArgTopK(k) => Some(k),
            ReduceOperationConfig::TopK(k) => Some(k),
            _ => None,
        },
    )?;

    launch_reduce::<R>(client, input, output, axis, strategy, dtypes, operation)
}

/// Reduce the given `axis` of `input`, writing the values into `values` and their indices
/// into `indices` from a **single** kernel launch.
///
/// Callers wanting both halves of a top-k would otherwise launch the reduce twice, once as
/// `TopK` and once as `ArgTopK`, and discard half of each result, even though one reduction
/// already computes both. This runs it once.
///
/// `values` and `indices` must have the same shape (the `input` shape with `axis` set to
/// `k`) and the same strides, since both outputs are written through one layout.
/// Only [`ReduceOperationConfig::TopK`] and [`ReduceOperationConfig::ArgTopK`] are accepted,
/// and they behave identically here since both halves are written regardless; any other
/// operation returns [`ReduceError::IndicesUnsupported`].
///
/// The plain [`reduce`] entrypoint is unaffected and still writes a single output.
#[allow(clippy::too_many_arguments)]
pub fn reduce_with_indices<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    values: TensorBinding<R>,
    indices: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
    operation: ReduceOperationConfig,
    dtypes: ReduceWithIndicesDtypes,
) -> Result<(), ReduceError> {
    let k = match operation {
        ReduceOperationConfig::TopK(k) | ReduceOperationConfig::ArgTopK(k) => k,
        other => {
            return Err(ReduceError::IndicesUnsupported {
                operation: operation_name(&other),
            });
        }
    };

    validate_axis(input.shape.len(), axis)?;
    validate_shapes(&input.shape, &values.shape, axis, Some(k))?;

    if indices.shape.as_slice() != values.shape.as_slice() {
        return Err(ReduceError::MismatchIndicesShape {
            values_shape: values.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
        });
    }

    // The two outputs must also share a layout: vectorization and write positions are
    // derived from the values tensor alone, so indices with different strides would be
    // written as if they had the values layout, producing silently wrong data.
    if indices.strides != values.strides {
        return Err(ReduceError::MismatchIndicesStrides {
            values_strides: values.strides.to_vec(),
            indices_strides: indices.strides.to_vec(),
        });
    }

    launch_reduce_with_indices::<R>(client, input, values, indices, axis, strategy, dtypes, k)
}

fn operation_name(operation: &ReduceOperationConfig) -> &'static str {
    match operation {
        ReduceOperationConfig::Sum => "Sum",
        ReduceOperationConfig::Prod => "Prod",
        ReduceOperationConfig::Mean => "Mean",
        ReduceOperationConfig::MaxAbs => "MaxAbs",
        ReduceOperationConfig::ArgMax => "ArgMax",
        ReduceOperationConfig::ArgMin => "ArgMin",
        ReduceOperationConfig::Max => "Max",
        ReduceOperationConfig::Min => "Min",
        ReduceOperationConfig::ArgTopK(_) => "ArgTopK",
        ReduceOperationConfig::TopK(_) => "TopK",
        ReduceOperationConfig::Any => "Any",
        ReduceOperationConfig::All => "All",
    }
}

// Check that the given axis is less than the rank of the input.
fn validate_axis(rank: usize, axis: usize) -> Result<(), ReduceError> {
    if axis > rank {
        return Err(ReduceError::InvalidAxis { axis, rank });
    }
    Ok(())
}

// Check that the output shape match the input shape with the given axis set to 1.
fn validate_shapes(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    k: Option<usize>,
) -> Result<(), ReduceError> {
    let mut expected_shape = input_shape.to_vec();
    let k = k.unwrap_or(1);

    if expected_shape[axis] < k {
        return Err(ReduceError::ReduceAxisTooSmall {
            axis_length: expected_shape[axis],
            k,
        });
    }

    expected_shape[axis] = k;
    if output_shape != expected_shape {
        return Err(ReduceError::MismatchOutputShape {
            expected_shape,
            output_shape: output_shape.to_vec(),
        });
    }
    Ok(())
}
