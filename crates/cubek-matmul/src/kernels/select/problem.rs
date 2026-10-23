//! **What the family's selector is asked about** — the product's dimensions,
//! the rows a memory-bound kernel would stream, and what each operand is.
//!
//! Every kernel has a problem of its own, holding what it reads: the routines'
//! in the seam's `problem` and each case's folder. This one holds only what the cross-kernel
//! choice turns on.

use cubecl::quant::scheme::QuantScheme;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use serde::{Deserialize, Serialize};

use cubecl::zspace::Shape;

/// The problem as the selector sees it, read off the operands' metadata.
///
/// Two of the fields are the *product*'s dimensions and two are the
/// *routing*'s, and they are not interchangeable: the rules read `k`, `n` and
/// [`streamed_rows`](Self::streamed_rows), while `m` and `batches` describe the shape a
/// caller passed and are what a refusal names back at them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulProblem {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Product of the broadcast batch extents (`1` when unbatched, and always
    /// `1` for a quantized operand, whose rows the caller merges into one).
    pub batches: usize,
    /// The lhs rows a memory-bound kernel would stream, or `None` where
    /// none of them serves this product. [`new`](Self::new) is where the number comes from,
    /// and every candidate counts the same way.
    ///
    /// Not `m`. Every memory-bound candidate merges the left-hand side's leading
    /// extents and reads the rhs once, so routing on the trailing extent
    /// sends a batched product to a table whose every candidate declines it,
    /// and the answer to a table that declines everything is the fallback
    /// rather than a kernel.
    pub streamed_rows: Option<usize>,
    /// What each input is: float, or packed. The output is float.
    pub operands: MatmulOperands<OperandKind>,
}

impl MatmulProblem {
    /// The selector's view of `lhs · rhs`, from each operand's logical shape and what it is
    /// alone: a storage-tiled operand stores each matrix dim as two physical dims, and the
    /// product's extents are the logical ones.
    pub fn new(
        (lhs_shape, lhs): (&Shape, OperandKind),
        (rhs_shape, rhs): (&Shape, OperandKind),
    ) -> Self {
        let rank = lhs_shape.rank();
        let batches = lhs_shape.as_slice()[..rank - 2]
            .iter()
            .zip(rhs_shape.as_slice()[..rank - 2].iter())
            .map(|(l, r)| *l.max(r))
            .product();
        Self {
            m: lhs_shape[rank - 2],
            k: lhs_shape[rank - 1],
            n: rhs_shape[rank - 1],
            batches,
            streamed_rows: Self::streamed_rows_of(lhs_shape, rhs_shape),
            operands: MatmulOperands {
                lhs,
                rhs,
                out: OperandKind::Float,
            },
        }
    }

    /// Each packed input and the scheme it is packed under.
    pub fn packed_inputs(&self) -> impl Iterator<Item = (MatmulOperand, QuantScheme)> + '_ {
        MatmulOperand::INPUTS
            .into_iter()
            .filter_map(|input| match self.operands.get(input) {
                OperandKind::Quantized(scheme) => Some((input, scheme)),
                OperandKind::Float => None,
            })
    }

    /// The lhs rows a memory-bound kernel would stream for `x @ w`, or
    /// `None` where none of them serves the product at all.
    ///
    /// Every one of them merges `x`'s leading extents into one row count and reads the
    /// rhs exactly once, so this — not `x`'s trailing extent — is the number
    /// they all work from, and the number routing has to ask. Reading the trailing
    /// extent instead calls `[64, 8, k] @ [1, k, n]` an 8-row product and sends a
    /// 512-row one to a table whose every candidate declines it.
    ///
    /// `None` is a batched rhs, which is one rhs per product where these
    /// kernels stream a single one, or a left-hand side whose feature extent is not the
    /// rhs's reduction: neither merges.
    fn streamed_rows_of(x_shape: &Shape, w_shape: &Shape) -> Option<usize> {
        let w_rank = w_shape.rank();
        let d_in = w_shape[w_rank - 2];
        let unbatched = w_shape.as_slice()[..w_rank - 2]
            .iter()
            .all(|&extent| extent == 1);
        let reduces = x_shape[x_shape.rank() - 1] == d_in;
        (unbatched && reduces && d_in > 0).then(|| x_shape.num_elements() / d_in)
    }
}

/// What an operand is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperandKind {
    Float,
    /// Quantized under `scheme`. Whether the kernels stream it
    /// ([`matmul_supported`](crate::kernels::select::packed::matmul_supported)) and whether the
    /// kernel's plan serves the
    /// problem on the device are the selector's to ask.
    Quantized(QuantScheme),
}
