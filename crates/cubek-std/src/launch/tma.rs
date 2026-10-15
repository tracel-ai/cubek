//! Helpers for building TMA (Tensor Memory Accelerator) descriptors.

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::TensorMapMeta;
pub use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides, shape, strides};

use cubek_tile::launch::TmaLanding;

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
    tma_operand_typed(
        binding,
        batches,
        layout,
        box_shape,
        remap_storage_for_tma(storage_ty),
        swizzle,
    )
}

/// [`tma_operand`] for a bulk-store target: the descriptor keeps `storage_ty` as it is, since a
/// tf32 descriptor rounds an f32 value to tf32 on its way out.
pub fn tma_store_operand(
    binding: TensorBinding,
    batches: usize,
    layout: MatrixLayout,
    box_shape: (usize, usize),
    storage_ty: ElemType,
    swizzle: TensorMapSwizzle,
) -> (TensorMapArg<Tiled>, bool) {
    tma_operand_typed(binding, batches, layout, box_shape, storage_ty, swizzle)
}

fn tma_operand_typed(
    binding: TensorBinding,
    batches: usize,
    layout: MatrixLayout,
    box_shape: (usize, usize),
    elem_ty: ElemType,
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
    let meta = tma_meta_tiled(Metadata::new(shape, strides), tile_size, elem_ty, swizzle);
    let arg = TensorMapArg {
        tensor: binding.into_tensor_arg(),
        metadata: meta,
        _kind: PhantomData,
    };
    (arg, transposed)
}

/// A storage-tiled operand's tensor map, its box `box_shape` landing as `landing` says.
/// `binding` is the operand as stored, `[.., R/tr, C/tc, tr, tc]`, its storage tile the
/// innermost pair. Where [`tma_operand`] collapses a binding to the descriptor's
/// `(batch, rows, cols)` and may swap the inner pair, this one keeps the stored rank: the tile is
/// stored a row at a time, so there is nothing to transpose.
///
/// - [`TmaLanding::Tiles`]: the descriptor is the binding as it is, and the box is one whole
///   storage tile, one contiguous run.
/// - [`TmaLanding::Rows`]: the descriptor is the binding as `[R/tr, tr, C/tc, tc]`, the tiles' rows
///   nested inside the tiles' columns, and a box of whole tiles both ways lands its rows one after
///   another, each a run of its tiles' rows, as a row-major operand's box would land them.
///
/// # Panics
///
/// A binding whose tile is not stored a row at a time: the descriptor takes the innermost dim to
/// step by one and each row to follow the last, so any other tile would load in silence as the
/// wrong values. Under [`TmaLanding::Tiles`], a box that is not the binding's tile; under
/// [`TmaLanding::Rows`], a binding with batch dims or a box that is not whole tiles along its
/// rows, which would land part of a tile's rows, nor whole tiles or a part dividing one along its
/// columns. The routine's own checks refuse these first.
pub fn tma_operand_tiled(
    binding: TensorBinding,
    box_shape: (usize, usize),
    landing: TmaLanding,
    storage_ty: ElemType,
    swizzle: TensorMapSwizzle,
) -> TensorMapArg<Tiled> {
    let rank = binding.shape.len();
    let tile = (binding.shape[rank - 2], binding.shape[rank - 1]);
    let inner = (binding.strides[rank - 2], binding.strides[rank - 1]);
    assert_eq!(
        inner,
        (tile.1, 1),
        "tma_operand_tiled: a box is one contiguous run only for a tile stored a row at a time, \
         which steps ({}, 1); this one steps {inner:?}",
        tile.1
    );
    let (metadata, box_dims) = match landing {
        TmaLanding::Tiles => {
            assert_eq!(
                box_shape, tile,
                "tma_operand_tiled: the binding is stored in {tile:?} tiles, not the {box_shape:?} \
                 asked for"
            );
            // Unit on every outer dim, the whole tile on the inner pair.
            let mut dims = vec![1usize; rank];
            dims[rank - 2] = tile.0;
            dims[rank - 1] = tile.1;
            let metadata = Metadata::new(binding.shape.clone(), binding.strides.clone());
            (metadata, Shape::from(dims))
        }
        TmaLanding::Rows => {
            assert_eq!(
                rank, 4,
                "tma_operand_tiled: a box landing in rows is over (R/tr, C/tc, tr, tc), unbatched"
            );
            let (box_rows, box_cols) = box_shape;
            // Along its columns a box is whole tiles, or a part of one tile's run dividing it.
            let (tiles_across, run) = match box_cols.is_multiple_of(tile.1) {
                true => (box_cols / tile.1, tile.1),
                false => (1, box_cols),
            };
            assert!(
                box_rows.is_multiple_of(tile.0) && tile.1.is_multiple_of(run),
                "tma_operand_tiled: a box of {box_shape:?} is not whole {tile:?} tiles along its \
                 rows, and whole tiles or a part dividing one along its columns"
            );
            let (shape, strides) = (&binding.shape, &binding.strides);
            let metadata = Metadata::new(
                shape![shape[0], shape[2], shape[1], shape[3]],
                strides![strides[0], strides[2], strides[1], strides[3]],
            );
            let box_dims = shape![box_rows / tile.0, tile.0, tiles_across, run];
            (metadata, box_dims)
        }
    };
    let meta = tma_meta_tiled(
        metadata,
        box_dims,
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

    /// A weight tiled to a 32x64 stage: `[256, 512]` stored as `[8, 8, 32, 64]`.
    fn tiled() -> TensorBinding {
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
            tiled(),
            (32, 64),
            TmaLanding::Tiles,
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
    fn a_descriptor_landing_in_rows_nests_the_tiles_rows_inside_their_columns() {
        let arg = tma_operand_tiled(
            tiled(),
            (64, 128),
            TmaLanding::Rows,
            f32::elem_type_native(),
            TensorMapSwizzle::None,
        );
        // `[R/tr, C/tc, tr, tc]` restated as `[R/tr, tr, C/tc, tc]`: the same memory.
        assert_eq!(&arg.metadata.metadata.shape()[..], &[8, 32, 8, 64]);
        assert_eq!(
            &arg.metadata.metadata.strides()[..],
            &[64 * 32 * 8, 64, 64 * 32, 1]
        );
        // Two tiles down, two across, each whole.
        let TensorMapFormat::Tiled(TiledArgs { tile_size }) = &arg.metadata.format else {
            panic!("expected a tiled tensor map");
        };
        assert_eq!(&tile_size[..], &[2, 32, 2, 64]);
    }

    #[test]
    #[should_panic(expected = "stored in (32, 64) tiles, not the (64, 64) asked for")]
    fn a_stored_descriptor_refuses_a_tile_the_buffer_does_not_hold() {
        tma_operand_tiled(
            tiled(),
            (64, 64),
            TmaLanding::Tiles,
            f32::elem_type_native(),
            TensorMapSwizzle::None,
        );
    }
}
