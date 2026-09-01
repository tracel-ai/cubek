use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::TensorBinding,
    zspace::{Shape, Strides},
};
use cubek_std::InputBinding;

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    strategy::Strategy,
};

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_ref_with_options(
        strategy,
        client,
        lhs,
        rhs,
        out,
        dtypes,
        LaunchOptions::default(),
    )
}

/// Rewrites applied to the problem before the strategy sees it. Every option is off
/// by default, so [`launch_ref`] behaves exactly as it did without them.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchOptions {
    /// Run a batched matmul whose right-hand side is the same matrix for every batch
    /// (`[b, m, k] × [1, k, n]`, or a stride-0 batch on the rhs) as one
    /// `[b·m, k] × [k, n]` product instead of `b` separate `[m, k] × [k, n]` ones.
    ///
    /// The rows of a matmul are independent, so folding the batch into `m` is exact
    /// when the rhs is shared. It matters because the problem is otherwise classified
    /// by its last two dims alone: with `m = 1` (one token per sequence at decode
    /// time) every batch becomes its own vector-matrix product and re-reads the whole
    /// weight matrix, `b` times the traffic of the single GEMM.
    ///
    /// Only applied when it is free: both operands must be plain (not quantized)
    /// tensors, and the lhs batch dims must already sit contiguously in front of its
    /// rows so no copy is needed. Otherwise the problem is launched untouched.
    pub collapse_broadcast_rhs_batches: bool,
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
/// [`launch_ref`] with [`LaunchOptions`].
pub fn launch_ref_with_options<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
    options: LaunchOptions,
) -> Result<(), MatmulSetupError> {
    let (lhs, rhs, out) = if options.collapse_broadcast_rhs_batches {
        collapse_broadcast_rhs_batches(lhs, rhs, out)
    } else {
        (lhs, rhs, out)
    };
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}

/// See [`LaunchOptions::collapse_broadcast_rhs_batches`]. Returns the operands
/// unchanged whenever the fold does not apply.
fn collapse_broadcast_rhs_batches<R: Runtime>(
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
) -> (InputBinding<R>, InputBinding<R>, TensorBinding<R>) {
    let (InputBinding::Normal(lhs_data, lhs_dtype), InputBinding::Normal(rhs_data, rhs_dtype)) =
        (&lhs, &rhs)
    else {
        return (lhs, rhs, out);
    };

    let Some((lhs_shape, lhs_strides, rhs_shape, rhs_strides, out_shape, out_strides)) =
        collapsed_views(
            &lhs_data.shape,
            &lhs_data.strides,
            &rhs_data.shape,
            &rhs_data.strides,
            &out.shape,
            &out.strides,
        )
    else {
        return (lhs, rhs, out);
    };

    let with_view = |mut binding: TensorBinding<R>, shape, strides| {
        binding.shape = shape;
        binding.strides = strides;
        binding
    };
    (
        InputBinding::Normal(
            with_view(lhs_data.clone(), lhs_shape, lhs_strides),
            *lhs_dtype,
        ),
        InputBinding::Normal(
            with_view(rhs_data.clone(), rhs_shape, rhs_strides),
            *rhs_dtype,
        ),
        with_view(out, out_shape, out_strides),
    )
}

type Views = (Shape, Strides, Shape, Strides, Shape, Strides);

/// The shapes and strides of the folded problem, or `None` when the fold does not
/// apply: the rhs differs across batches, a batch dim of the lhs or the output is
/// not laid out right in front of its rows, or the ranks disagree.
fn collapsed_views(
    lhs_shape: &Shape,
    lhs_strides: &Strides,
    rhs_shape: &Shape,
    rhs_strides: &Strides,
    out_shape: &Shape,
    out_strides: &Strides,
) -> Option<Views> {
    let rank = lhs_shape.len();
    if rank < 3 || out_shape.len() != rank || rhs_shape.len() < 2 {
        return None;
    }
    // Every batch of the rhs is the same matrix: each batch dim is either absent
    // (extent 1) or broadcast (stride 0).
    let rhs_batches = rhs_shape.len() - 2;
    if !(0..rhs_batches).all(|d| rhs_shape[d] == 1 || rhs_strides[d] == 0) {
        return None;
    }

    let (lhs_shape, lhs_strides) = fold_batches_into_rows(lhs_shape, lhs_strides)?;
    let (out_shape, out_strides) = fold_batches_into_rows(out_shape, out_strides)?;
    if lhs_shape[rank - 2] != out_shape[rank - 2] {
        // An lhs with fewer batches than the output (the rhs carried the batch
        // extent) is a real broadcast of the lhs, not a fold.
        return None;
    }

    // The rhs keeps its rank; its batch dims just become extent 1 with the stride a
    // fresh contiguous tensor would have, so the layout still reads as one batch.
    let mut rhs_shape = rhs_shape.clone();
    let mut rhs_strides = rhs_strides.clone();
    let footprint = matrix_footprint(&rhs_shape, &rhs_strides);
    for d in 0..rhs_batches {
        rhs_shape[d] = 1;
        rhs_strides[d] = footprint;
    }

    Some((
        lhs_shape,
        lhs_strides,
        rhs_shape,
        rhs_strides,
        out_shape,
        out_strides,
    ))
}

/// The same tensor viewed with all its batch dims folded into the row dim, or
/// `None` when that view would need a copy (a batch stride that is not the extent
/// of everything inside it times that inner stride). Rank is kept: the batch dims
/// stay as extent 1 with the stride a fresh contiguous tensor would have.
fn fold_batches_into_rows(shape: &Shape, strides: &Strides) -> Option<(Shape, Strides)> {
    let rank = shape.len();
    let rows_dim = rank - 2;
    let mut rows = shape[rows_dim];
    let mut row_stride = strides[rows_dim];
    for d in (0..rows_dim).rev() {
        if shape[d] == 1 {
            continue;
        }
        if rows == 1 {
            // A single row has no stride of its own; the folded rows take this dim's.
            rows = shape[d];
            row_stride = strides[d];
            continue;
        }
        if strides[d] != rows * row_stride {
            return None;
        }
        rows *= shape[d];
    }

    let mut shape = shape.clone();
    let mut strides = strides.clone();
    shape[rows_dim] = rows;
    strides[rows_dim] = row_stride;
    let footprint = matrix_footprint(&shape, &strides);
    for d in 0..rows_dim {
        shape[d] = 1;
        strides[d] = footprint;
    }
    Some((shape, strides))
}

/// Number of elements spanned by the last two dims: the batch stride of a
/// contiguous tensor with this matrix layout.
fn matrix_footprint(shape: &Shape, strides: &Strides) -> usize {
    let rank = shape.len();
    (shape[rank - 2] * strides[rank - 2]).max(shape[rank - 1] * strides[rank - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn views(
        lhs: (&[usize], &[usize]),
        rhs: (&[usize], &[usize]),
        out: (&[usize], &[usize]),
    ) -> Option<Views> {
        let shape = |dims: &[usize]| dims.iter().copied().collect::<Shape>();
        collapsed_views(
            &shape(lhs.0),
            &Strides::new(lhs.1),
            &shape(rhs.0),
            &Strides::new(rhs.1),
            &shape(out.0),
            &Strides::new(out.1),
        )
    }

    /// The decode shape: 16 sequences of one token each through one weight matrix.
    #[test]
    fn decode_batch_folds_into_rows() {
        let (b, m, k, n) = (16, 1, 64, 32);
        let (lhs_shape, lhs_strides, rhs_shape, rhs_strides, out_shape, out_strides) = views(
            (&[b, m, k], &[k, k, 1]),
            (&[1, k, n], &[k * n, n, 1]),
            (&[b, m, n], &[n, n, 1]),
        )
        .expect("the fold applies");
        assert_eq!(lhs_shape.to_vec(), [1, b * m, k]);
        assert_eq!(lhs_strides.to_vec(), [b * k, k, 1]);
        assert_eq!(rhs_shape.to_vec(), [1, k, n]);
        assert_eq!(rhs_strides.to_vec(), [k * n, n, 1]);
        assert_eq!(out_shape.to_vec(), [1, b * m, n]);
        assert_eq!(out_strides.to_vec(), [b * n, n, 1]);
    }

    /// Several rows per batch fold too, and a stride-0 rhs batch counts as shared.
    #[test]
    fn multi_row_batches_and_stride_zero_rhs_fold() {
        let (b, m, k, n) = (4, 8, 64, 32);
        let (lhs_shape, _, rhs_shape, rhs_strides, out_shape, _) = views(
            (&[b, m, k], &[m * k, k, 1]),
            (&[b, k, n], &[0, n, 1]),
            (&[b, m, n], &[m * n, n, 1]),
        )
        .expect("the fold applies");
        assert_eq!(lhs_shape.to_vec(), [1, b * m, k]);
        assert_eq!(rhs_shape.to_vec(), [1, k, n]);
        assert_eq!(rhs_strides.to_vec(), [k * n, n, 1]);
        assert_eq!(out_shape.to_vec(), [1, b * m, n]);
    }

    /// A column-major lhs folds as long as its batches sit right after its rows.
    #[test]
    fn transposed_lhs_folds() {
        let (b, m, k, n) = (4, 8, 64, 32);
        let (lhs_shape, lhs_strides, ..) = views(
            (&[b, m, k], &[m, 1, b * m]),
            (&[1, k, n], &[k * n, n, 1]),
            (&[b, m, n], &[m * n, n, 1]),
        )
        .expect("the fold applies");
        assert_eq!(lhs_shape.to_vec(), [1, b * m, k]);
        assert_eq!(lhs_strides.to_vec(), [b * m * k, 1, b * m]);
    }

    #[test]
    fn leaves_other_problems_alone() {
        let (b, m, k, n) = (4, 8, 64, 32);
        // A different rhs per batch: not a shared matrix.
        assert!(
            views(
                (&[b, m, k], &[m * k, k, 1]),
                (&[b, k, n], &[k * n, n, 1]),
                (&[b, m, n], &[m * n, n, 1]),
            )
            .is_none()
        );
        // Padded lhs batches: folding would need a copy.
        assert!(
            views(
                (&[b, m, k], &[2 * m * k, k, 1]),
                (&[1, k, n], &[k * n, n, 1]),
                (&[b, m, n], &[m * n, n, 1]),
            )
            .is_none()
        );
        // The lhs itself is broadcast over the batch: the rhs carries the extent.
        assert!(
            views(
                (&[1, m, k], &[m * k, k, 1]),
                (&[b, k, n], &[0, n, 1]),
                (&[b, m, n], &[m * n, n, 1]),
            )
            .is_none()
        );
        // Nothing to fold without a batch dim.
        assert!(views((&[m, k], &[k, 1]), (&[k, n], &[n, 1]), (&[m, n], &[n, 1])).is_none());
    }
}
