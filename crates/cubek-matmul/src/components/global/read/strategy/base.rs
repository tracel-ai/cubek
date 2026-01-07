use crate::components::global::memory::GlobalIterator;
use crate::components::stage::TilingLayout;
use crate::components::stage::{StageMemoryConfig, SwizzleMode};
use crate::components::{global::GlobalReaderConfig, stage::StageConfig};
use crate::components::{global::SharedGlobalMatmulConfig, stage::StageFamily};
use crate::definition::{
    InvalidConfigError, MatmulElems, MatmulPrecision, MatmulProblem, MatrixLayout, StageIdent,
};
use cubecl::ir::{BarrierLevel, OpaqueType, SemanticType};
use cubecl::prelude::*;

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<EG: Numeric, ES: Numeric, TL: TilingLayout, S: SyncStrategy>:
    CubeType + Clone
{
    type Stage: StageFamily;

    /// Execute the `task_id`th loading task
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut <Self::Stage as StageFamily>::Stage<ES, TL>,
        barrier: &mut S::Barrier,
        #[comptime] config: GlobalReaderConfig,
    );

    /// Get the number of tasks
    fn task_count(this: &Self) -> comptime_type!(u32);
}

/// A synchronization strategy determines the type of synchronization object, how to create it and
/// how to synchronize on it.
/// The sync strategy must match the one on both the LHS and RHS loading strategy.
#[cube]
pub trait SyncStrategy {
    type Barrier: CubeType + Clone;
    fn create_barrier() -> Self::Barrier;
    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    );
}

/// Allows to verify configs are valid for a reader
pub trait LoadingValidation {
    /// Verify that configs are valid for a reader, otherwise return an error stating why
    fn validate_with_config(config: &GlobalReaderConfig) -> Result<(), InvalidConfigError>;

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError>;
}

/// Validates if async barrier instructions is available on the current device.
pub fn validate_async_barrier() -> Result<(), InvalidConfigError> {
    if !comptime::device_properties()
        .features
        .supports_type(OpaqueType::Barrier(BarrierLevel::Cube))
    {
        return Err(Box::new(
            "Async barrier instructions are not available on the current device",
        ));
    }

    Ok(())
}

/// Validates if async copy instructions is available on the current device.
pub fn validate_async_copy(
    dtype_global: &StorageType,
    dtype_stage: &StorageType,
) -> Result<(), InvalidConfigError> {
    if !comptime::device_properties().features.copy_async {
        return Err(Box::new(
            "Async copy instructions are not available on the current device",
        ));
    }

    if dtype_global.size() != dtype_stage.size() {
        return Err(Box::new(
            "Async copy requires stage and global types to be the same",
        ));
    }

    if matches!(dtype_global, StorageType::Packed(_, _))
        && !matches!(dtype_stage, StorageType::Packed(_, _))
    {
        return Err(Box::new(
            "Async copy doesn't support dequantizing on global read",
        ));
    }

    Ok(())
}

/// Validates if swizzling is disabled, for loaders that can't support it.
pub fn validate_noswizzle(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
    if config.swizzle != SwizzleMode::None {
        return Err(Box::new("This loader doesn't support swizzling"));
    }

    Ok(())
}

/// Validates if swizzling is valid with the line size, for sync readers that read in terms of full
/// lines
pub fn validate_swizzle_atom_size(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
    if config.swizzle == SwizzleMode::None {
        return Ok(());
    }

    let line_bytes = config.dtype.size() * config.line_size as usize;
    if line_bytes > config.swizzle.atom_size() {
        return Err(Box::new("Load atom can't be larger than swizzle atom"));
    }

    Ok(())
}

/// Validates if [tensor memory accelerator features](SemanticType::TensorMap) are available on the current
/// device.
pub fn validate_tma(
    smem_config: &StageMemoryConfig,
    global_dtype: &StorageType,
) -> Result<(), InvalidConfigError> {
    if !comptime::device_properties()
        .features
        .supports_type(SemanticType::TensorMap)
    {
        return Err(Box::new(
            "Tensor memory accelerator features are not available on the current device",
        ));
    }

    let stage_dtype = smem_config.dtype;

    if global_dtype.size() != stage_dtype.size() {
        return Err(Box::new(
            "TMA requires stage and global types to be the same",
        ));
    }

    if matches!(global_dtype, StorageType::Packed(_, _))
        && !matches!(stage_dtype, StorageType::Packed(_, _))
    {
        return Err(Box::new("TMA doesn't support dequantizing on global read"));
    }

    if matches!(smem_config.swizzle, SwizzleMode::None) {
        return Ok(());
    }

    let row_size = match smem_config.matrix_layout {
        MatrixLayout::RowMajor => smem_config.elements_per_stage_along_col(),
        MatrixLayout::ColMajor => smem_config.elements_per_stage_along_row(),
    };
    let row_bytes = row_size * global_dtype.size() as u32;

    // Slightly tighter than the actual requirements, but simple enough and is always followed by
    // selection. Getting illegal memory access if this isn't followed for some reason.
    if row_bytes as usize != smem_config.swizzle.span_size() {
        return Err(Box::new("Swizzling size must be equal to row size for TMA"));
    }

    Ok(())
}

pub fn validate_async_copy_with_problem(
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    ident: StageIdent,
) -> Result<(), InvalidConfigError> {
    let (strides, layout) = match ident {
        StageIdent::Lhs => (&problem.lhs_strides, &problem.lhs_layout),
        StageIdent::Rhs => (&problem.rhs_strides, &problem.rhs_layout),
        _ => unreachable!("Should be a loadable tensors"),
    };

    if stride_align_bits(strides, layout, &dtypes.global(ident.into())) < 4 {
        return Err(Box::new(
            "Async copy requires strides to be aligned to 16 bytes",
        ));
    }

    Ok(())
}

pub fn validate_tma_with_problem(
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    ident: StageIdent,
) -> Result<(), InvalidConfigError> {
    let (strides, layout) = match ident {
        StageIdent::Lhs => (&problem.lhs_strides, &problem.lhs_layout),
        StageIdent::Rhs => (&problem.rhs_strides, &problem.rhs_layout),
        _ => unreachable!("Should be a loadable tensors"),
    };

    if stride_align_bits(strides, layout, &dtypes.global(ident.into())) < 4 {
        return Err(Box::new("TMA requires strides to be aligned to 16 bytes"));
    }

    Ok(())
}

/// Defines the non-contiguous stride alignment in terms of powers of two
fn stride_align_bits(strides: &[usize], layout: &MatrixLayout, dtype: &StorageType) -> u32 {
    let exclude_dim = match layout {
        MatrixLayout::RowMajor => strides.len() - 1,
        MatrixLayout::ColMajor => strides.len() - 2,
    };
    strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_dim)
        .map(|(_, it)| (*it * dtype.size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(31)
}

/// Dummy trait implementation
pub struct NoLoadingValidation {}
impl LoadingValidation for NoLoadingValidation {
    fn validate_with_config(_config: &GlobalReaderConfig) -> Result<(), InvalidConfigError> {
        Ok(())
    }

    fn validate_with_problem(
        _problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        _ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Controls bounds checking for reader operations.
///
/// This **does not** disable tensor read bounds checks.
/// It only affects checks for whether the reader loads more data than allowed
/// at each global matmul iteration.
pub enum ReaderMode {
    /// Enforces compile-time validation of balanced workloads across units.
    /// Restricts valid combinations of tile shape, count, and line size.
    Strict,
    /// Inserts runtime checks only when an out-of-bounds access will occur.
    /// May reduce performance if workloads are imbalanced.
    #[default]
    Relaxed,
}
