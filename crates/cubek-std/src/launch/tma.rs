//! Helpers for building TMA (Tensor Memory Accelerator) descriptors.

use cubecl::ir::StorageType;
use cubecl::prelude::*;
use cubecl::server::TensorMapMeta;
pub use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};

use crate::MatrixLayout;

/// CUDA's TMA loads f32 as tf32 internally; remap explicitly so the descriptor matches.
pub fn remap_storage_for_tma(ty: StorageType) -> StorageType {
    if ty == f32::as_type_native_unchecked().storage_type() {
        tf32::as_type_native_unchecked().storage_type()
    } else {
        ty
    }
}

/// TMA assumes the last stride is contiguous and discards it. For ColMajor inputs we therefore
/// swap the inner two dims so the contiguous one ends up last. The tensor's own metadata stays
/// in its original layout — only the TMA descriptor sees the transposed form.
///
/// `shape` and `strides` may have different ranks (the matmul builder constructs them
/// transiently mismatched and aligns them afterwards). Each is swapped on its own inner pair.
///
/// Returns `true` if a swap occurred.
pub fn transpose_inner_for_tma(
    shape: &mut Shape,
    strides: &mut Strides,
    layout: MatrixLayout,
) -> bool {
    if matches!(layout, MatrixLayout::ColMajor) {
        let s_rank = shape.num_dims();
        let t_rank = strides.rank();
        shape.swap(s_rank - 1, s_rank - 2);
        strides.swap(t_rank - 1, t_rank - 2);
        true
    } else {
        false
    }
}

/// Build a tiled [`TensorMapMeta`] with the defaults shared by every current call site
/// (no interleave, no prefetch, OOB-fill = zero, elem_stride = `[1; rank]`).
pub fn tma_meta_tiled(
    metadata: Metadata,
    tile_size: Shape,
    storage_ty: StorageType,
    swizzle: TensorMapSwizzle,
) -> TensorMapMeta {
    let rank = metadata.rank();
    TensorMapMeta {
        format: TensorMapFormat::Tiled(TiledArgs { tile_size }),
        metadata,
        elem_stride: Strides::new(&vec![1; rank]),
        interleave: TensorMapInterleave::None,
        swizzle,
        prefetch: TensorMapPrefetch::None,
        oob_fill: OobFill::Zero,
        storage_ty,
    }
}
