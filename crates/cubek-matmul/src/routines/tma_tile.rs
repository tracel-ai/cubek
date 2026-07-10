//! Building a matmul operand's TMA descriptor, then loading it via the tile API.
//!
//! The descriptor geometry (3-D collapse, f32→tf32 remap, col-major transpose) is matmul +
//! cubek-std knowledge — `tma_meta_tiled` lives in cubek-std, which the tile engine doesn't
//! depend on — so it can't be generic. This thin builder does that geometry, then hands the
//! tensor map to [`TmaArgLaunch::tensor_map`].

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::zspace::{Strides, metadata::Metadata, shape, strides};
use cubek_std::MatrixLayout;
use cubek_std::launch::tma::{remap_storage_for_tma, tma_meta_tiled, transpose_inner_for_tma};
use cubek_tile::{Space, TmaArgLaunch};

/// Load one operand of logical `(batches, rows, cols)` and major order `layout` as a TMA tile
/// argument. Collapses to the 3-D `(batch, rows, cols)` the descriptor expects, transposes the
/// inner pair for a col-major operand (TMA discards the last stride), and builds the tensor map
/// with `box_shape` — the logical `(rows, cols)` one bulk copy moves, i.e. the smem stage.
/// `space` is the operand's already-projected tile space.
pub fn operand_tma<R: Runtime, E: Numeric>(
    binding: TensorBinding<R>,
    (batches, rows, cols): (usize, usize, usize),
    layout: MatrixLayout,
    box_shape: (usize, usize),
    storage_ty: StorageType,
    space: Space,
) -> TmaArgLaunch<E, R> {
    let mut shape = shape![batches, rows, cols];
    let rank = binding.strides.len();
    let mut strides: Strides = if rank > 2 {
        binding.strides[rank - 3..].into()
    } else {
        strides![binding.strides[0], binding.strides[1]]
    };

    let transposed = transpose_inner_for_tma(&mut shape, &mut strides, layout);
    // Re-insert the batch stride after the (possible) inner swap so it stays outermost.
    if strides.len() == 2 {
        let stride = strides[0];
        strides.insert(0, stride);
    }

    // The box in descriptor order; one bulk copy fills one smem stage.
    let (box_rows, box_cols) = box_shape;
    let tile_size = match transposed {
        true => shape![1, box_cols, box_rows],
        false => shape![1, box_rows, box_cols],
    };
    let meta = tma_meta_tiled(
        Metadata::new(shape, strides),
        tile_size,
        remap_storage_for_tma(storage_ty),
        TensorMapSwizzle::None,
    );
    let tensor_map = TensorMapArg {
        tensor: binding.into_tensor_arg(),
        metadata: meta,
        _kind: PhantomData,
    };

    TmaArgLaunch::tensor_map(
        tensor_map,
        space,
        (batches as u32, rows as u32, cols as u32),
        (box_rows as u32, box_cols as u32),
        transposed,
    )
}
