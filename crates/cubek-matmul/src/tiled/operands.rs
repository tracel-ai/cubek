//! The axes every tiled routine builds its space over.

use cubecl::{prelude::TensorBinding, zspace::metadata::Metadata};
use cubek_tile::{Axis, Level, Partitioning, Space};

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
/// storage-tiled: one nesting on both matrix dims, whose tile fragments are the buffer's two
/// innermost dims. `None` for a plain buffer.
///
/// # Errors
///
/// A tiling this routine cannot read: a batch dim stored in fragments, or the matrix dims
/// stored to different depths or deeper than one nesting.
pub(crate) fn storage_tile(
    binding: &TensorBinding,
    name: &str,
) -> Result<Option<(usize, usize)>, MatmulSetupError> {
    if !binding.tiling.is_tiled() {
        return Ok(None);
    }
    let rank = binding.shape.len();
    let logical_rank = binding
        .tiling
        .logical_rank(rank)
        .map_err(|e| MatmulSetupError::InvalidConfig(Box::new(format!("{name}: {e:?}"))))?;
    let fragments = binding.tiling.fragments(logical_rank);
    let (batches, matrix) = fragments.split_at(logical_rank - 2);
    if batches.iter().any(|&n| n != 1) || matrix != [2, 2] {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "{name}: a storage-tiled operand stores both matrix dims one nesting deep and its \
             batch dims plain, got {fragments:?} fragments"
        ))));
    }
    Ok(Some((binding.shape[rank - 2], binding.shape[rank - 1])))
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
    Err(MatmulSetupError::InvalidConfig(Box::new(format!(
        "{name} is stored in {tile:?} storage tiles, which is the tile of no level of this \
         routine's nest; pack the tensor to one of its tiles"
    ))))
}

#[cfg(test)]
mod tests {
    use cubecl::{prelude::TensorBinding, zspace::Tiling};
    use cubek_tile::Level;

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
        let levels = vec![
            Level::cubes(&[(M, 16), (N, 32)]),
            Level::planes(&[(M, 8), (N, 8)]),
            Level::walk(&[(K, 4)]),
        ];
        let cube_tile = binding(&[2, 4, 16, 32], Tiling::new(&[2, 2]).unwrap());
        validate_stored_tile(&cube_tile, "out", &space, &levels, (M, N)).unwrap();
        let no_level = binding(&[4, 4, 16, 16], Tiling::new(&[2, 2]).unwrap());
        assert!(validate_stored_tile(&no_level, "out", &space, &levels, (M, N)).is_err());
        let plain = binding(&[64, 64], Tiling::UNTILED);
        validate_stored_tile(&plain, "out", &space, &levels, (M, N)).unwrap();
    }
}
