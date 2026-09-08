//! The axes every tiled routine builds its space over.

use cubecl::{prelude::TensorBinding, zspace::metadata::Metadata};
use cubek_tile::Axis;

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
