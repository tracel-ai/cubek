use crate::{
    definition::{MatmulElems, MatmulProblem, StageIdent},
    multi_level::{
        components::{
            global::{GlobalReaderConfig, SharedGlobalMatmulConfig, memory::GlobalIterator},
            stage::StageFamily,
        },
        definition::MatmulTypes,
        stage::{StageMemoryConfig, SwizzleMode},
        tile::TilingLayout,
    },
};
use cubecl::{
    ir::{DeviceProperties, OpaqueType},
    prelude::*,
};
use cubek_std::{InvalidConfigError, MatrixLayout, launch::tma::stride_align_bits};

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<
    EG: Numeric,
    NG: Size,
    ES: Numeric,
    NS: Size,
    TL: TilingLayout,
    S: SyncStrategy,
>: CubeType<ExpandType: Clone> + Clone
{
    type Stage: StageFamily;

    /// Execute the `task_id`th loading task
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut <Self::Stage as StageFamily>::Stage<ES, NS, TL>,
        barrier: &S::Barrier,
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
    type Barrier: CubeType<ExpandType: Clone> + Clone;
    fn create_barrier() -> Self::Barrier;
    fn sync<MP: MatmulTypes>(barrier: &Self::Barrier, #[comptime] config: SharedGlobalMatmulConfig);
}

/// Allows to verify configs are valid for a reader
pub trait LoadingValidation {
    /// Verify that configs are valid for a reader, otherwise return an error stating why
    fn validate_with_config(
        device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError>;

    fn validate_with_problem(
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError>;
}

/// Validates if async barrier instructions is available on the current device.
pub fn validate_async_barrier(device_props: &DeviceProperties) -> Result<(), InvalidConfigError> {
    if !device_props.features.supports_type(OpaqueType::Barrier) {
        return Err(Box::new(
            "Async barrier instructions are not available on the current device",
        ));
    }

    Ok(())
}

/// Validates if async copy instructions is available on the current device.
pub fn validate_async_copy(
    device_props: &DeviceProperties,
    dtype_global: &ElemType,
    dtype_stage: &ElemType,
) -> Result<(), InvalidConfigError> {
    if !device_props.features.copy_async {
        return Err(Box::new(
            "Async copy instructions are not available on the current device",
        ));
    }

    if dtype_global.size() != dtype_stage.size() {
        return Err(Box::new(
            "Async copy requires stage and global types to be the same",
        ));
    }

    // if matches!(dtype_global, StorageType::Packed(_, _))
    //     && !matches!(dtype_stage, StorageType::Packed(_, _))
    // {
    //     return Err(Box::new(
    //         "Async copy doesn't support dequantizing on global read",
    //     ));
    // }

    Ok(())
}

/// Validates if swizzling is valid with the vector size, for sync readers that read in terms of full
/// vectors
pub fn validate_swizzle_atom_size(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
    if config.swizzle == SwizzleMode::None {
        return Ok(());
    }

    let vector_bytes = config.dtype.size() * config.vector_size as usize;
    if vector_bytes > config.swizzle.atom_size() {
        return Err(Box::new("Load atom can't be larger than swizzle atom"));
    }

    Ok(())
}

/// The most elements a tensor map's box may span along one dimension (`boxDim` of
/// `cuTensorMapEncodeTiled`).
pub const TMA_MAX_BOX_DIM: u32 = 256;

/// The box, in logical `(rows, cols)` of the operand, that a TMA reader loads a stage of
/// `stage` elements with, the stage made of tiles of `tile` elements: the whole stage along the
/// strided axis, and along the contiguous one a single tile, since without swizzle bank
/// conflicts cap the box at a one-tile-wide strip, or the whole stage when swizzled.
pub fn tma_box(
    stage: (u32, u32),
    tile: (u32, u32),
    layout: MatrixLayout,
    swizzle: SwizzleMode,
) -> (u32, u32) {
    match swizzle {
        SwizzleMode::None => match layout {
            MatrixLayout::RowMajor => (stage.0, tile.1),
            MatrixLayout::ColMajor => (tile.0, stage.1),
        },
        _ => stage,
    }
}

/// [`tma_box`] for the stage `smem_config` describes.
pub fn tma_stage_box(smem_config: &StageMemoryConfig) -> (u32, u32) {
    tma_box(
        (
            smem_config.elements_per_stage_along_row(),
            smem_config.elements_per_stage_along_col(),
        ),
        (
            smem_config.elements_per_tile_along_row,
            smem_config.elements_per_tile_along_col,
        ),
        smem_config.matrix_layout,
        smem_config.swizzle,
    )
}

/// Validates that the box a TMA reader loads the stage `smem_config` describes with is one a
/// tensor map can be encoded with. The encoding happens at launch, so a box past the limit is
/// otherwise a launch that fails after the configuration was accepted.
pub fn validate_tma_box(smem_config: &StageMemoryConfig) -> Result<(), InvalidConfigError> {
    let (rows, cols) = tma_stage_box(smem_config);
    if rows > TMA_MAX_BOX_DIM || cols > TMA_MAX_BOX_DIM {
        return Err(Box::new(format!(
            "TMA box {rows}x{cols} exceeds the {TMA_MAX_BOX_DIM} elements a tensor map allows \
             along each dimension"
        )));
    }

    Ok(())
}

/// Validates if [tensor memory accelerator features](OpaqueType::TensorMap) are available on the current
/// device, and that the stage can be loaded with boxes a tensor map accepts.
pub fn validate_tma(
    device_props: &DeviceProperties,
    smem_config: &StageMemoryConfig,
    global_dtype: &ElemType,
) -> Result<(), InvalidConfigError> {
    if !device_props.features.supports_type(OpaqueType::TensorMap) {
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

    // if matches!(global_dtype, StorageType::Packed(_, _))
    //     && !matches!(stage_dtype, StorageType::Packed(_, _))
    // {
    //     return Err(Box::new("TMA doesn't support dequantizing on global read"));
    // }

    validate_tma_box(smem_config)?;

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
    let is_quantized = match ident {
        StageIdent::Lhs => problem.lhs_scheme.is_some(),
        StageIdent::Rhs => problem.rhs_scheme.is_some(),
        StageIdent::Acc | StageIdent::Out => false,
    };

    if is_quantized {
        return Err(Box::new(
            "Async copy doesn't support dequantizing on global read",
        ));
    }

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
    let is_quantized = match ident {
        StageIdent::Lhs => problem.lhs_scheme.is_some(),
        StageIdent::Rhs => problem.rhs_scheme.is_some(),
        StageIdent::Acc | StageIdent::Out => false,
    };

    if is_quantized {
        return Err(Box::new("TMA doesn't support dequantizing on global read"));
    }

    let (strides, layout) = match ident {
        StageIdent::Lhs => (&problem.lhs_strides, &problem.lhs_layout),
        StageIdent::Rhs => (&problem.rhs_strides, &problem.rhs_layout),
        _ => unreachable!("Should be a loadable tensors"),
    };

    if stride_align_bits(strides, layout, &dtypes.global(ident.into())) < 4 {
        return Err(Box::new("TMA requires strides to be aligned to 16 bytes"));
    }

    if problem.lhs_batches != problem.rhs_batches
        && problem.lhs_batches.iter().product::<usize>() != 1
        && problem.rhs_batches.iter().product::<usize>() != 1
    {
        return Err(Box::new(
            "TMA doesn't support mixing broadcast and non-broadcast dims",
        ));
    }

    Ok(())
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Controls bounds checking for reader operations.
///
/// This **does not** disable tensor read bounds checks.
/// It only affects checks for whether the reader loads more data than allowed
/// at each global matmul iteration.
pub enum ReaderMode {
    /// Enforces compile-time validation of balanced workloads across units.
    /// Restricts valid combinations of tile shape, count, and vector size.
    Strict,
    /// Inserts runtime checks only when an out-of-bounds access will occur.
    /// May reduce performance if workloads are imbalanced.
    #[default]
    Relaxed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    /// A bf16 stage of `rows x cols` tiles of `tile` elements, one partition along the columns.
    fn stage(
        tile: (u32, u32),
        tiles: (u32, u32),
        layout: MatrixLayout,
        swizzle: SwizzleMode,
    ) -> StageMemoryConfig {
        StageMemoryConfig {
            num_planes: 4,
            elements_per_tile_along_row: tile.0,
            elements_per_tile_along_col: tile.1,
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: tiles.1,
            partitions_per_stage_along_row: tiles.0,
            partitions_per_stage_along_col: 1,
            vector_size: 8,
            matrix_layout: layout,
            swizzle,
            num_stages: 1,
            dtype: ElemType::Float(FloatKind::BF16),
        }
    }

    #[test]
    fn the_box_spans_the_stage_along_the_strided_axis_and_a_tile_along_the_other() {
        let row_major = stage((32, 16), (16, 2), MatrixLayout::RowMajor, SwizzleMode::None);
        assert_eq!(tma_stage_box(&row_major), (512, 16));
        let col_major = stage((16, 32), (2, 16), MatrixLayout::ColMajor, SwizzleMode::None);
        assert_eq!(tma_stage_box(&col_major), (16, 512));
        let swizzled = stage((16, 16), (8, 4), MatrixLayout::RowMajor, SwizzleMode::B128);
        assert_eq!(tma_stage_box(&swizzled), (128, 64));
    }

    /// The stage cubek-matmul infers for a bf16 `[6144 x 3584] x [3584 x 1024]` with the cmma
    /// tile `32x8x16`: sixteen planes of one tile row each, 512 rows the tensor map would have
    /// to span in one box.
    #[test]
    fn a_stage_taller_than_a_box_can_span_is_refused() {
        let lhs = stage((32, 16), (16, 2), MatrixLayout::RowMajor, SwizzleMode::None);
        let err = validate_tma_box(&lhs).expect_err("a 512-row box cannot be encoded");
        assert!(err.to_string().contains("512x16"), "{err}");
    }

    #[test]
    fn a_stage_a_box_can_span_is_accepted() {
        for config in [
            stage((32, 16), (8, 2), MatrixLayout::RowMajor, SwizzleMode::None),
            stage((16, 16), (16, 4), MatrixLayout::RowMajor, SwizzleMode::B128),
            stage((16, 32), (2, 8), MatrixLayout::ColMajor, SwizzleMode::None),
        ] {
            assert!(
                validate_tma_box(&config).is_ok(),
                "{:?}",
                tma_stage_box(&config)
            );
        }
    }
}
