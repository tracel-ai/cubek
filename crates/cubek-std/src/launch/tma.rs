//! Helpers for building TMA (Tensor Memory Accelerator) descriptors.

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::TensorMapMeta;
pub use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides, shape, strides};

use crate::MatrixLayout;

/// Defines the non-contiguous stride alignment in terms of powers of two
pub fn stride_align_bits(strides: &[usize], layout: &MatrixLayout, dtype: &ElemType) -> u32 {
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

/// CUDA's TMA loads f32 as tf32 internally; remap explicitly so the descriptor matches.
pub fn remap_storage_for_tma(ty: ElemType) -> ElemType {
    if ty == f32::elem_type_native() {
        tf32::elem_type_native()
    } else {
        ty
    }
}

/// TMA assumes the last stride is contiguous and discards it. For ColMajor inputs we therefore
/// swap the inner two dims so the contiguous one ends up last. The tensor's own metadata stays
/// in its original layout: only the TMA descriptor sees the transposed form.
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

/// One matmul-style operand's tiled tensor map. Collapses the binding to the 3-D
/// `(batch, rows, cols)` the descriptor expects, swaps the inner pair for a col-major
/// operand (TMA discards the last stride) and keeps the batch stride outermost.
/// `box_shape` is the box one bulk copy moves, in logical `(rows, cols)`; it rides the
/// same swap. Returns the arg plus whether the swap occurred.
pub fn tma_operand(
    binding: TensorBinding,
    batches: usize,
    layout: MatrixLayout,
    box_shape: (usize, usize),
    storage_ty: ElemType,
    swizzle: TensorMapSwizzle,
) -> (TensorMapArg<Tiled>, bool) {
    let rank = binding.shape.len();
    let mut shape = shape![batches, binding.shape[rank - 2], binding.shape[rank - 1]];
    let mut strides: Strides = if rank > 2 {
        binding.strides[rank - 3..].into()
    } else {
        strides![binding.strides[0], binding.strides[1]]
    };
    let transposed = transpose_inner_for_tma(&mut shape, &mut strides, layout);
    // Re-insert the batch stride after the (possible) swap so it stays outermost.
    if strides.len() == 2 {
        let stride = strides[0];
        strides.insert(0, stride);
    }

    let (box_rows, box_cols) = box_shape;
    let tile_size = match transposed {
        true => shape![1, box_cols, box_rows],
        false => shape![1, box_rows, box_cols],
    };
    let meta = tma_meta_tiled(
        Metadata::new(shape, strides),
        tile_size,
        remap_storage_for_tma(storage_ty),
        swizzle,
    );
    let arg = TensorMapArg {
        tensor: binding.into_tensor_arg(),
        metadata: meta,
        _kind: PhantomData,
    };
    (arg, transposed)
}

/// A storage-tiled operand's tensor map: the descriptor is the physical
/// `[.., R/tr, C/tc, tr, tc]` the data is already stored in, and the box is one whole storage
/// tile. Where [`tma_operand`] collapses a binding to the descriptor's `(batch, rows, cols)`
/// and may swap the inner pair, this one takes the binding as it is: the storage tile is the
/// innermost pair and therefore the fastest-varying, so one box is one contiguous run and there
/// is nothing to transpose.
///
/// # Panics
///
/// A binding whose innermost two dims are not `tile`, which is what packing produced and what
/// the routine's own storage-tile check enforces.
pub fn tma_operand_tiled(
    binding: TensorBinding,
    tile: (usize, usize),
    storage_ty: ElemType,
    swizzle: TensorMapSwizzle,
) -> TensorMapArg<Tiled> {
    let rank = binding.shape.len();
    let stored = (binding.shape[rank - 2], binding.shape[rank - 1]);
    assert_eq!(
        stored, tile,
        "tma_operand_tiled: the binding is stored in {stored:?} tiles, not the {tile:?} asked for"
    );
    // One box per storage tile: unit on every outer dim, the whole tile on the inner pair.
    let mut dims = vec![1usize; rank];
    dims[rank - 2] = tile.0;
    dims[rank - 1] = tile.1;
    let box_shape: Shape = dims.into();
    let meta = tma_meta_tiled(
        Metadata::new(binding.shape.clone(), binding.strides.clone()),
        box_shape,
        remap_storage_for_tma(storage_ty),
        swizzle,
    );
    TensorMapArg {
        tensor: binding.into_tensor_arg(),
        metadata: meta,
        _kind: PhantomData,
    }
}

/// Build a tiled [`TensorMapMeta`] with the defaults shared by every current call site
/// (no interleave, no prefetch, OOB-fill = zero, elem_stride = `[1; rank]`).
pub fn tma_meta_tiled(
    metadata: Metadata,
    tile_size: Shape,
    elem_ty: ElemType,
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
        elem_ty,
    }
}

#[cfg(test)]
mod tests {
    use cubecl::zspace::Tiling;

    use super::*;

    /// A weight packed to a 32x64 stage: `[256, 512]` stored as `[8, 8, 32, 64]`.
    fn packed() -> TensorBinding {
        let client = cubecl::test_device().client();
        let shape = shape![8, 8, 32, 64];
        let strides = strides![64 * 32 * 8, 64 * 32, 64, 1];
        TensorBinding {
            handle: client.empty(256 * 512 * 4).binding(),
            strides,
            shape,
            tiling: Tiling::new(&[2, 2]).unwrap(),
        }
    }

    #[test]
    fn a_stored_descriptor_keeps_the_rank_and_boxes_one_storage_tile() {
        let arg = tma_operand_tiled(
            packed(),
            (32, 64),
            f32::elem_type_native(),
            TensorMapSwizzle::None,
        );
        // The descriptor is the buffer as it is stored, not collapsed to (batch, row, col).
        assert_eq!(&arg.metadata.metadata.shape()[..], &[8, 8, 32, 64]);
        assert_eq!(
            &arg.metadata.metadata.strides()[..],
            &[64 * 32 * 8, 64 * 32, 64, 1]
        );
        // One box is one storage tile: unit outside, the whole tile on the inner pair, which is
        // the fastest-varying pair, so the box is one contiguous run.
        let TensorMapFormat::Tiled(TiledArgs { tile_size }) = &arg.metadata.format else {
            panic!("expected a tiled tensor map");
        };
        assert_eq!(&tile_size[..], &[1, 1, 32, 64]);
    }

    #[test]
    #[should_panic(expected = "stored in (32, 64) tiles, not the (64, 64) asked for")]
    fn a_stored_descriptor_refuses_a_tile_the_buffer_does_not_hold() {
        tma_operand_tiled(
            packed(),
            (64, 64),
            f32::elem_type_native(),
            TensorMapSwizzle::None,
        );
    }
}
