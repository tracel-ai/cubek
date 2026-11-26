//! The reduce operation, [`Reduce`]: built from a [`ReduceOpStrategy`], it launches the kernel the
//! strategy picks on the tile routines.

use crate::ReduceError;
use crate::kernels::{
    ReduceElements, ReduceOpStrategy, ReduceTensors, ReduceWorkspace, launch_by_strategy,
};
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubek_microkernel::reduce::ReduceOperation;

/// A reduction of one tensor along one axis, run by the kernel its strategy picks, with the
/// scratch its caller gives the kernel whose last cube merges a row split across cubes.
///
/// # Assumptions
///
/// - The input has fewer than 256 axes.
/// - The outputs alias neither the input nor each other.
///
/// # Guarantees
///
/// - Each output is written in the layout of the buffer its caller gives.
#[derive(Debug, Clone)]
pub struct Reduce {
    strategy: ReduceOpStrategy,
    workspace: Option<ReduceWorkspace>,
}

impl Reduce {
    /// The reduction whose kernel `strategy` picks, given no scratch: the last-arrival hypercube
    /// declines it.
    pub fn new(strategy: ReduceOpStrategy) -> Self {
        Self {
            strategy,
            workspace: None,
        }
    }

    /// This reduction given `workspace`, the last-arrival hypercube's scratch, which cubek never
    /// allocates. The hypercube declines a reduction given fewer bytes than
    /// [`ReduceWorkspace::bytes_for`] states where its launch needs them.
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkspace`]: its counters at zero when first given, its launches one
    ///   after another, its buffers aliasing no operand nor each other.
    pub fn with_workspace(&self, workspace: ReduceWorkspace) -> Self {
        Self {
            workspace: Some(workspace),
            ..self.clone()
        }
    }

    /// Reduce dim `axis` of `input` into `output` by `operation`, in `elements`.
    /// `output` has `input`'s shape with `axis` one wide, `k` wide for a top-k.
    ///
    /// # Errors
    ///
    /// [`ReduceError::InvalidAxis`] where `axis` is not a dim of `input`,
    /// [`ReduceError::EmptyTopK`] where a top-k keeps `k = 0` values,
    /// [`ReduceError::ReduceAxisTooSmall`] where the reduced dim is empty, or shorter than the `k` a
    /// top-k keeps, [`ReduceError::MismatchOutputShape`] where `output`'s shape is not the one
    /// above, and [`ReduceError::Declined`] with the kernel's decline where the kernel the strategy
    /// picks cannot run the problem on the client's device: a
    /// [`Specific`](ReduceOpStrategy::Specific) one, or the unit kernel the rule falls back to. A
    /// [`Decline::Broken`](cubecl::tile::launch::Decline::Broken) in it is a bug of the routine
    /// that ran, not a problem it cannot run.
    ///
    /// # Panics
    ///
    /// Where an Assumption of [`Reduce`] is broken: an input of 256 axes or more.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn execute(
        &self,
        client: &Client,
        input: TensorBinding,
        output: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Result<(), ReduceError> {
        self.launch(ReduceTensors::checked(
            client, input, output, axis, operation, elements,
        )?)
    }

    /// [`execute`](Self::execute), the values written to `values` in `elements.out` and their
    /// indices to `indices` in `index_element`, from one launch. `indices` has `values`' shape, and
    /// each is written at its own strides. `Max`, `Min` and `TopK` write the same pair as `ArgMax`,
    /// `ArgMin` and `ArgTopK`.
    ///
    /// # Errors
    ///
    /// Every [`ReduceError`].
    ///
    /// # Panics
    ///
    /// As [`execute`](Self::execute).
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn execute_with_indices(
        &self,
        client: &Client,
        input: TensorBinding,
        values: TensorBinding,
        indices: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        index_element: ElemType,
    ) -> Result<(), ReduceError> {
        let tensors = ReduceTensors::checked_with_indices(
            client,
            input,
            values,
            indices,
            axis,
            operation,
            elements,
            index_element,
        )?;
        self.launch(tensors)
    }

    /// `tensors`, checked to lie in the reduction's Domain
    /// ([`ReduceTensors::checked`]), reduced by the kernel the strategy picks, given this
    /// reduction's workspace where it has one.
    ///
    /// # Errors
    ///
    /// [`ReduceError::Declined`] with the kernel's decline where the kernel the strategy picks
    /// cannot run the problem on the client's device, or a
    /// [`Decline::Broken`](cubecl::tile::launch::Decline::Broken) where it breaks what it
    /// prepared.
    ///
    /// # Panics
    ///
    /// As [`execute`](Self::execute).
    pub fn launch(&self, tensors: ReduceTensors) -> Result<(), ReduceError> {
        launch_by_strategy(self.strategy, given(self, tensors))
    }
}

/// `tensors` given the workspace `reduction` has, where it has one.
fn given(reduction: &Reduce, tensors: ReduceTensors) -> ReduceTensors {
    match &reduction.workspace {
        Some(workspace) => tensors.with_workspace(workspace.clone()),
        None => tensors,
    }
}

/// The operation that writes `operation`'s values with their indices: the value and its index
/// accumulate into the same carrier, so `Max` is `ArgMax` with both halves written.
///
/// # Errors
///
/// [`ReduceError::IndicesUnsupported`] for an operation whose values carry no index.
pub(crate) fn indexed(operation: ReduceOperation) -> Result<ReduceOperation, ReduceError> {
    match operation {
        ReduceOperation::Max | ReduceOperation::ArgMax => Ok(ReduceOperation::ArgMax),
        ReduceOperation::Min | ReduceOperation::ArgMin => Ok(ReduceOperation::ArgMin),
        ReduceOperation::TopK { k } | ReduceOperation::ArgTopK { k } => {
            Ok(ReduceOperation::ArgTopK { k })
        }
        ReduceOperation::Sum => Err(unindexed_operation_named("Sum")),
        ReduceOperation::Prod => Err(unindexed_operation_named("Prod")),
        ReduceOperation::Mean => Err(unindexed_operation_named("Mean")),
        ReduceOperation::MaxAbs => Err(unindexed_operation_named("MaxAbs")),
        ReduceOperation::Any => Err(unindexed_operation_named("Any")),
        ReduceOperation::All => Err(unindexed_operation_named("All")),
    }
}

/// Why [`Reduce::execute_with_indices`] refuses the operation `operation_name`.
fn unindexed_operation_named(operation_name: &'static str) -> ReduceError {
    ReduceError::IndicesUnsupported {
        operation: operation_name,
    }
}

/// Whether `axis` is a dim of a tensor of rank `rank`.
pub(crate) fn validate_axis(rank: usize, axis: usize) -> Result<(), ReduceError> {
    if axis >= rank {
        return Err(ReduceError::InvalidAxis { axis, rank });
    }
    Ok(())
}

/// Whether `output_shape` is `input_shape` with `axis` `written_per_row` wide, and the reduced dim
/// at least that long.
pub(crate) fn validate_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    written_per_row: usize,
) -> Result<(), ReduceError> {
    if written_per_row == 0 {
        return Err(ReduceError::EmptyTopK);
    }
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

/// Whether the indices' shape is the values': an index for every value.
pub(crate) fn validate_indices(
    values_shape: &[usize],
    indices_shape: &[usize],
) -> Result<(), ReduceError> {
    if indices_shape != values_shape {
        return Err(ReduceError::MismatchIndicesShape {
            values_shape: values_shape.to_vec(),
            indices_shape: indices_shape.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceKernel;
    use crate::kernels::testing::zeros;

    const ROW_COUNT: usize = 4;
    const ROW_VALUE_COUNT: usize = 8;
    const AXIS: usize = 1;

    #[test]
    fn a_top_k_keeping_nothing_is_outside_the_domain() {
        let result = validate_shape(&[ROW_COUNT, ROW_VALUE_COUNT], &[ROW_COUNT, 0], AXIS, 0);
        assert!(matches!(result, Err(ReduceError::EmptyTopK)), "{result:?}");
    }

    #[test]
    fn a_top_k_keeping_more_than_a_row_is_outside_the_domain() {
        let k = ROW_VALUE_COUNT + 1;
        let result = validate_shape(&[ROW_COUNT, ROW_VALUE_COUNT], &[ROW_COUNT, k], AXIS, k);
        assert!(
            matches!(result, Err(ReduceError::ReduceAxisTooSmall { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn an_output_not_one_wide_along_the_axis_is_outside_the_domain() {
        let result = validate_shape(&[ROW_COUNT, ROW_VALUE_COUNT], &[ROW_COUNT, 2], AXIS, 1);
        assert!(
            matches!(result, Err(ReduceError::MismatchOutputShape { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn an_axis_past_the_rank_is_outside_the_domain() {
        let rank = 2;
        let result = validate_axis(rank, rank);
        assert!(
            matches!(result, Err(ReduceError::InvalidAxis { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn indices_of_another_shape_are_outside_the_domain() {
        let result = validate_indices(&[ROW_COUNT, 1], &[1, ROW_COUNT]);
        assert!(
            matches!(result, Err(ReduceError::MismatchIndicesShape { .. })),
            "{result:?}"
        );
    }

    /// Checks that `operation` is refused under its own `name`.
    fn has_no_index_to_write(operation: ReduceOperation, name: &str) {
        let result = indexed(operation);
        assert!(
            matches!(result, Err(ReduceError::IndicesUnsupported { operation }) if operation == name),
            "{result:?}"
        );
    }

    #[test]
    fn a_sum_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::Sum, "Sum");
    }

    #[test]
    fn a_prod_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::Prod, "Prod");
    }

    #[test]
    fn a_mean_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::Mean, "Mean");
    }

    #[test]
    fn a_max_abs_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::MaxAbs, "MaxAbs");
    }

    #[test]
    fn an_any_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::Any, "Any");
    }

    #[test]
    fn an_all_has_no_index_to_write() {
        has_no_index_to_write(ReduceOperation::All, "All");
    }

    #[test]
    fn a_max_with_indices_writes_an_argmax() {
        assert!(matches!(
            indexed(ReduceOperation::Max),
            Ok(ReduceOperation::ArgMax)
        ));
    }

    #[test]
    fn a_min_with_indices_writes_an_argmin() {
        assert!(matches!(
            indexed(ReduceOperation::Min),
            Ok(ReduceOperation::ArgMin)
        ));
    }

    #[test]
    fn a_top_k_with_indices_writes_an_arg_top_k() {
        assert!(matches!(
            indexed(ReduceOperation::TopK { k: 3 }),
            Ok(ReduceOperation::ArgTopK { k: 3 })
        ));
    }

    #[test]
    fn a_well_shaped_call_is_inside_the_domain() {
        assert!(validate_shape(&[ROW_COUNT, ROW_VALUE_COUNT], &[ROW_COUNT, 1], AXIS, 1).is_ok());
    }

    const UNIT: Reduce = Reduce {
        strategy: ReduceOpStrategy::Specific(ReduceKernel::Unit),
        workspace: None,
    };

    /// The workspace of a reduction no test here launches.
    fn workspace() -> ReduceWorkspace {
        let client = cubecl::test_device().client();
        ReduceWorkspace {
            partials: client.empty(64),
            partial_indices: client.empty(32),
            arrivals: client.empty(8),
        }
    }

    #[test]
    fn a_new_reduction_is_given_no_workspace() {
        let reduction = Reduce::new(ReduceOpStrategy::default());
        let tensors =
            ReduceTensors::zeros(&[ROW_COUNT, ROW_VALUE_COUNT], AXIS, ReduceOperation::Sum);
        assert_eq!(given(&reduction, tensors).problem().partials, None);
    }

    #[test]
    fn a_reductions_workspace_is_given_to_its_tensors() {
        let reduction = UNIT.with_workspace(workspace());
        let tensors =
            ReduceTensors::zeros(&[ROW_COUNT, ROW_VALUE_COUNT], AXIS, ReduceOperation::Sum);
        assert!(given(&reduction, tensors).problem().partials.is_some());
    }

    /// Two output buffers of a row a cell, and the zeros they are reduced from.
    fn zeros_and_outputs() -> [TensorBinding; 3] {
        let client = cubecl::test_device().client();
        [
            zeros(&client, &[ROW_COUNT, ROW_VALUE_COUNT]),
            zeros(&client, &[ROW_COUNT, 1]),
            zeros(&client, &[ROW_COUNT, 1]),
        ]
    }

    fn elements() -> ReduceElements {
        let f32 = f32::elem_type_native();
        ReduceElements::for_operation(ReduceOperation::Max, f32, None)
    }

    #[test]
    fn a_call_inside_the_domain_runs() {
        let [input, output, _] = zeros_and_outputs();
        let result = UNIT.execute(
            &cubecl::test_device().client(),
            input,
            output,
            AXIS,
            ReduceOperation::Max,
            elements(),
        );
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn a_call_outside_the_domain_errs() {
        let [input, output, _] = zeros_and_outputs();
        let result = UNIT.execute(
            &cubecl::test_device().client(),
            input,
            output,
            AXIS + 1,
            ReduceOperation::Max,
            elements(),
        );
        assert!(
            matches!(result, Err(ReduceError::InvalidAxis { .. })),
            "{result:?}"
        );
    }

    #[test]
    fn a_call_with_indices_inside_the_domain_runs() {
        let [input, values, indices] = zeros_and_outputs();
        let result = UNIT.execute_with_indices(
            &cubecl::test_device().client(),
            input,
            values,
            indices,
            AXIS,
            ReduceOperation::Max,
            elements(),
            u32::elem_type_native(),
        );
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn a_call_with_indices_outside_the_domain_errs() {
        let [input, values, indices] = zeros_and_outputs();
        let result = UNIT.execute_with_indices(
            &cubecl::test_device().client(),
            input,
            values,
            indices,
            AXIS,
            ReduceOperation::Sum,
            elements(),
            u32::elem_type_native(),
        );
        assert!(
            matches!(result, Err(ReduceError::IndicesUnsupported { .. })),
            "{result:?}"
        );
    }
}
