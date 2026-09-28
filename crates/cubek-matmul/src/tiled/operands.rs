//! The axes every tiled routine builds its space over.

use cubecl::{
    prelude::TensorBinding,
    zspace::{Tiling, metadata::Metadata},
};
use cubek_tile::{
    Axis, Geometry, Level, Partitioning, Space,
    layout::{Layout, StorageTiling},
};

use crate::definition::MatmulSetupError;

// The matmul tile axes, shared by every tiled routine that lays out its space over `(m, n, k)`
// plus batches. `M`/`N`/`K` are the two matrix dims and the contraction; batch axes follow.
pub(crate) const M: Axis = Axis(0);
pub(crate) const N: Axis = Axis(1);
pub(crate) const K: Axis = Axis(2);

/// The axis for output batch dimension `i` (outermost is `0`).
pub(crate) fn batch_axis(i: usize) -> Axis {
    Axis(3 + i as u8)
}

/// The space a routine's levels are stated over before any problem: every axis dynamic, so the
/// counts a shape decides print as unknown and the tiling itself still reads. The axis order is
/// the launch's — batch axes first, then the matrix dims and the contraction — since that order
/// is what a [`Region`](cubek_tile::Region)'s coordinates come in.
///
/// Nothing may ask this space for a grid: [`Partitioning::cube_count`] needs an extent the
/// launch has not stamped yet. Printing is what it is for, and only tests print one today. It
/// widens when a caller does.
#[cfg(test)]
pub(crate) fn form_space(batches: usize) -> Space {
    let axes: Vec<Axis> = (0..batches).map(batch_axis).chain([M, N, K]).collect();
    Space::dynamic(&axes)
}

/// What a partitioning over these axes prints as
/// ([`Partitioning::labelled`]): an [`Axis`] is an index, and only the routine that assigned it
/// knows what it stands for. Read off the space rather than off a batch list, so any caller
/// holding one can name its axes.
pub(crate) fn labels(space: &Space) -> Vec<(Axis, &'static str)> {
    const BATCHES: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];
    let mut batches = 0;
    space
        .axes()
        .map(|axis| {
            let name = if axis == M {
                "m"
            } else if axis == N {
                "n"
            } else if axis == K {
                "k"
            } else {
                batches += 1;
                BATCHES[batches - 1]
            };
            (axis, name)
        })
        .collect()
}

/// Logical `(batches, rows, cols)` off a matrix operand's binding, which may be storage-tiled:
/// the tensor states how it is stored, so its own metadata folds the fragments back and no
/// routine has to be told.
pub(crate) fn logical_dims(binding: &TensorBinding) -> (Vec<usize>, usize, usize) {
    let logical = Metadata::new(binding.shape.clone(), binding.strides.clone())
        .with_tiling(binding.tiling)
        .expect("a binding's tiling describes its own rank")
        .logical_shape()
        .expect("a binding's tiling describes its own rank");
    let dims = logical.as_slice();
    let split = dims.len() - 2;
    (dims[..split].to_vec(), dims[split], dims[split + 1])
}

/// The storage tile a matrix operand's binding is stored in, `(rows, cols)`, when it is
/// storage-tiled: each matrix dim over its grid's count, the grid being each dim's coarsest
/// piece, which a tiling lists first. `None` for a plain buffer.
///
/// The tile may be stored in finer pieces than a routine reads (a vector read, a packed word
/// first), and it is read the same as long as they fuse back into it a row at a time: the stored
/// layout has to [refine](cubek_tile::Layout::refines) a row-first `rows x cols` tile, and the
/// order the tiles themselves follow one another in is the strides', whatever it is.
///
/// # Errors
///
/// A tiling this routine cannot read: a batch dim stored in pieces, or a tile that is not stored
/// a row at a time.
pub(crate) fn storage_tile(
    binding: &TensorBinding,
    name: &str,
) -> Result<Option<(usize, usize)>, MatmulSetupError> {
    let refused = |why: String| MatmulSetupError::InvalidConfig(Box::new(format!("{name}: {why}")));
    if !binding.tiling.is_tiled() {
        return Ok(None);
    }
    let rank = binding.shape.len();
    let logical_rank = binding
        .tiling
        .logical_rank(rank)
        .map_err(|e| refused(format!("{e:?}")))?;
    let fragments = binding.tiling.fragments(logical_rank);
    let batches = logical_rank - 2;
    if fragments[..batches].iter().any(|&n| n != 1) {
        return Err(refused(format!(
            "batch dims are stored plain, got {fragments:?} pieces per dim"
        )));
    }
    let (_, rows, cols) = logical_dims(binding);
    let tile = (
        rows / binding.shape[batches],
        cols / binding.shape[batches + 1],
    );
    let labels = StorageTiling::stored(binding.tiling, 2, rank).order(&[TILE_ROWS, TILE_COLS]);
    let stored = Layout::of(&Geometry::from(binding), &labels);
    let row_first = Layout::wanted(&[(TILE_COLS, tile.1), (TILE_ROWS, tile.0)]);
    stored.refines(&row_first).map_err(|why| {
        refused(format!(
            "it is not stored in {}x{} tiles a row at a time: {why}",
            tile.0, tile.1
        ))
    })?;
    Ok(Some(tile))
}

/// The labels a stored matrix's two dims take while its layout is read.
const TILE_ROWS: Axis = Axis(0);
const TILE_COLS: Axis = Axis(1);

/// A storage-tiled binding as a routine that moves whole tiles sees it: the same buffer, its
/// tile's pieces fused into one row-first `rows x cols` tile, `[.., R/tr, C/tc, tr, tc]` over its
/// grid's own strides. A plain binding is returned as it is.
///
/// # Errors
///
/// What [`storage_tile`] refuses.
pub(crate) fn fused_to_tile(
    binding: TensorBinding,
    name: &str,
) -> Result<TensorBinding, MatmulSetupError> {
    let Some((tr, tc)) = storage_tile(&binding, name)? else {
        return Ok(binding);
    };
    let rank = binding.shape.len();
    let logical_rank = binding
        .tiling
        .logical_rank(rank)
        .expect("storage_tile read it");
    let batches = logical_rank - 2;
    let mut shape: Vec<usize> = binding.shape[..batches + 2].to_vec();
    let mut strides: Vec<usize> = binding.strides[..batches + 2].to_vec();
    shape.extend([tr, tc]);
    strides.extend([tc, 1]);
    let fragments: Vec<usize> = (0..batches).map(|_| 1).chain([2, 2]).collect();
    let mut fused = binding;
    fused.shape = shape.into();
    fused.strides = strides.into();
    fused.tiling = Tiling::new(&fragments).expect("one level over two dims fits any tiling");
    Ok(fused)
}

/// Refuse, on the host, a storage-tiled matrix operand whose tile is the tile of none of
/// `levels` on `(rows, cols)`: the launch matches it on a worker thread, where a refusal
/// reaches no caller. A plain operand passes.
///
/// # Errors
///
/// A tiling [`storage_tile`] cannot read, or a tile no level cuts to.
#[allow(clippy::result_large_err)]
pub(crate) fn validate_stored_tile(
    binding: &TensorBinding,
    name: &str,
    space: &Space,
    levels: &[Level],
    (rows, cols): (Axis, Axis),
) -> Result<(), MatmulSetupError> {
    let Some(tile) = storage_tile(binding, name)? else {
        return Ok(());
    };
    let cuts_to = |i: usize| {
        let leaf = Partitioning::new(space.clone(), levels[..=i].to_vec()).leaf();
        (leaf.extent(rows), leaf.extent(cols)) == tile
    };
    if (0..levels.len()).any(cuts_to) {
        return Ok(());
    }
    let partitioning = Partitioning::new(space.clone(), levels.to_vec());
    Err(MatmulSetupError::InvalidConfig(Box::new(format!(
        "{name} is stored in {tile:?} storage tiles, which is the tile of no level of this \
         routine's nest; tile the tensor to one of its tiles\n\n{}",
        partitioning.table(&labels(space))
    ))))
}

#[cfg(test)]
mod tests {
    use cubecl::{prelude::TensorBinding, zspace::Tiling};

    use super::*;

    fn binding(shape: &[usize], tiling: Tiling) -> TensorBinding {
        let client = cubecl::test_device().client();
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let len: usize = shape.iter().product();
        TensorBinding {
            handle: client.empty(len * 4).binding(),
            strides: strides.into(),
            shape: shape.to_vec().into(),
            tiling,
        }
    }

    #[test]
    fn a_stored_tile_that_is_a_level_passes_and_one_that_is_none_is_refused() {
        let space = Space::new(&[(M, 64), (N, 64), (K, 32)]);
        let levels = cubek_tile::Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[K])
            .planes(&[(M, 2), (N, 4)])
            .cubes(&[M, N])
            .build();
        let cube_tile = binding(&[2, 4, 16, 32], Tiling::new(&[2, 2]).unwrap());
        validate_stored_tile(&cube_tile, "out", &space, &levels, (M, N)).unwrap();
        let no_level = binding(&[4, 4, 16, 16], Tiling::new(&[2, 2]).unwrap());
        assert!(validate_stored_tile(&no_level, "out", &space, &levels, (M, N)).is_err());
        let plain = binding(&[64, 64], Tiling::UNTILED);
        validate_stored_tile(&plain, "out", &space, &levels, (M, N)).unwrap();
    }
}
