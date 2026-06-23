//! Building a matmul operand's TMA descriptor, then loading it via the shared tile API.
//!
//! TMA's descriptor (the `tile_size` box, `swizzle`, f32→tf32 remap, col-major transpose) is
//! matmul + cubek-std knowledge — and `tma_meta_tiled` lives in cubek-std, which the tile engine
//! doesn't depend on — so it can't be generic. This thin builder does that geometry, then the
//! actual tile construction is [`TileArgLaunch::tma`] in cubek-tile.

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::zspace::{Shape, Strides, metadata::Metadata, shape, strides};
use cubek_std::launch::tma::{remap_storage_for_tma, tma_meta_tiled, transpose_inner_for_tma};
use cubek_std::{MatrixLayout, stage::SwizzleMode};
use cubek_tile::{Space, TileArgLaunch};

/// Load one operand of logical `(batches, rows, cols)` and major order `layout` as a TMA tile.
/// Collapses to the 3-D `(batch, rows, cols)` the descriptor expects, transposes the inner pair for
/// a col-major operand (TMA discards the last stride), builds the tensor-map, and hands it to
/// [`TileArgLaunch::tma`]. `space` is the operand's already-projected tile space.
#[allow(clippy::too_many_arguments)]
pub fn operand_tma<R: Runtime, E: Numeric, V: Size>(
    binding: TensorBinding<R>,
    batches: usize,
    rows: usize,
    cols: usize,
    layout: MatrixLayout,
    swizzle: SwizzleMode,
    tile_size: Shape,
    storage_ty: StorageType,
    space: Space,
) -> TileArgLaunch<E, V, R> {
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

    let meta = tma_meta_tiled(
        Metadata::new(shape, strides),
        tile_size,
        remap_storage_for_tma(storage_ty),
        swizzle.into(),
    );
    let tensor_map = TensorMapArg {
        tensor: binding.into_tensor_arg(),
        metadata: meta,
        _kind: PhantomData,
    };

    let (box_rows, box_cols) = match transposed {
        true => (cols as u32, rows as u32),
        false => (rows as u32, cols as u32),
    };
    TileArgLaunch::tma(
        tensor_map,
        space,
        (batches as u32, box_rows, box_cols),
        transposed,
    )
}
