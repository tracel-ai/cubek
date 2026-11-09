//! The tile routines behind the same entry as [`reduce`](crate::reduce): what a caller switches to
//! once the routines have been measured against today's kernels, and what the suite runs on both.

use crate::kernels::{
    CubeReduceRoutine, HypercubeReduceRoutine, PlaneReduceRoutine, ReduceElements, ReduceRoutine,
    ReduceSplitExtents, ReduceTensors, UnitReduceRoutine,
};
use crate::launch::RoutineStrategy;
use crate::{
    ReduceDtypes, ReduceError, ReduceOperationConfig, ReduceStrategy, ReduceWithIndicesDtypes,
};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// [`reduce`](crate::reduce) on the tile routines: the routine of the scope `strategy` names
/// (unit, plane or cube) at the first of its settings this device prepares the problem at; for
/// the cube, the hypercube where too few rows leave the device idle. The strategy's blueprint
/// and vectorization are today's kernels' and are not read.
///
/// # Errors
///
/// As [`reduce`](crate::reduce)'s validation, and [`ReduceError::Validation`] with the routine's
/// reason where it declines the problem.
pub fn reduce_on_tiles(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
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
    launched_by_scope(strategy, &tensors)
}

/// [`reduce_with_indices`](crate::reduce_with_indices) on the tile routines: the values to
/// `values` and their indices to `indices` from one launch, by the routine of the scope `strategy`
/// names, as [`reduce_on_tiles`] picks it.
///
/// # Errors
///
/// As [`reduce_with_indices`](crate::reduce_with_indices)'s validation, and
/// [`ReduceError::Validation`] with the routine's reason where it declines the problem.
#[allow(clippy::too_many_arguments)]
pub fn reduce_with_indices_on_tiles(
    client: &Client,
    input: TensorBinding,
    values: TensorBinding,
    indices: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
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
    launched_by_scope(strategy, &tensors)
}

/// `tensors` reduced by the routine of the scope `strategy` names.
fn launched_by_scope(strategy: ReduceStrategy, tensors: &ReduceTensors) -> Result<(), ReduceError> {
    let launched = match strategy.routine {
        RoutineStrategy::Unit(_) => launch_first_served(&UnitReduceRoutine, tensors),
        RoutineStrategy::Plane(_) => launch_first_served(&PlaneReduceRoutine, tensors),
        RoutineStrategy::Cube(_) => launch_cube_or_split(tensors),
    };
    launched.map_err(|why| ReduceError::Validation {
        details: match why {
            Refusal::Declined { why } => why,
            _ => "the tile reduce routine cannot launch this problem on this device",
        },
    })
}

/// `tensors` reduced by `routine` at the first of its settings this device prepares them at.
fn launch_first_served<R: ReduceRoutine<M>, M>(
    routine: &R,
    tensors: &ReduceTensors,
) -> Result<(), Refusal> {
    let hw = Hardware::from_client(tensors.client());
    let (blueprint, grid) = routine.prepare_first_served(&hw, tensors.problem())?;
    routine.launch(tensors, blueprint, grid);
    Ok(())
}

/// `tensors` reduced a box of rows a cube where the boxes fill the device, otherwise split along
/// the reduced dim by the hypercube: at the first of its settings, the longest parts first, whose
/// cubes fill the device, or at the one with the most cubes where none does. Where the device does
/// not say how wide it is, or every row is a single part, the cube routine.
fn launch_cube_or_split(tensors: &ReduceTensors) -> Result<(), Refusal> {
    let hw = Hardware::from_client(tensors.client());
    let problem = tensors.problem();
    let routine = HypercubeReduceRoutine;
    let mut most_cubes = None;
    for settings in routine.settings(&hw) {
        let Ok((blueprint, grid)) = routine.prepare(&hw, problem, &settings) else {
            continue;
        };
        let units = (blueprint.microkernel.lanes * blueprint.microkernel.planes) as u32;
        let Some(fill) = hw.cubes_to_fill(units).map(|fill| fill as usize) else {
            break;
        };
        let boxes = ReduceSplitExtents::row_boxes(problem, blueprint.width);
        let parts = ReduceSplitExtents::new(problem, &blueprint.partitioning, blueprint.width).parts
            as usize;
        if boxes >= fill {
            break;
        }
        if parts < 2 {
            continue;
        }
        let cubes = boxes * parts;
        if cubes >= fill {
            most_cubes = Some((cubes, blueprint, grid));
            break;
        }
        if most_cubes.as_ref().is_none_or(|(most, _, _)| cubes > *most) {
            most_cubes = Some((cubes, blueprint, grid));
        }
    }
    match most_cubes {
        Some((_, blueprint, grid)) => {
            routine.launch(tensors, blueprint, grid);
            Ok(())
        }
        None => launch_first_served(&CubeReduceRoutine, tensors),
    }
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
