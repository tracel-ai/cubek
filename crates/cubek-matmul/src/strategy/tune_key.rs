use cubecl::{
    tune::anchor,
    zspace::{Shape, Strides},
    {AutotuneKey, Runtime, quant::scheme::QuantScheme},
    {client::ComputeClient, ir::StorageType},
};
use cubek_std::MatmulProblemSize;
use serde::{Deserialize, Serialize};

use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};

use crate::definition::MatmulKind;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    pub definition: MatmulProblemDefinition,
    pub analysis: MatmulAutotuneAnalysis,
}

/// Maximum factor relevant for strides. Currently set to 2^10 because that's 128-byte swizzle's
/// repeat number, so it's the largest align that can have performance impacts.
const MAX_STRIDE_FACTOR: u32 = 10;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MatmulProblemDefinition {
    #[autotune(anchor)]
    pub m: usize,
    #[autotune(anchor)]
    pub n: usize,
    #[autotune(anchor)]
    pub k: usize,
    pub lhs_pow2_factor: u8,
    /// Power of two that lhs strides are aligned to
    pub lhs_stride_factor: u8,
    pub rhs_pow2_factor: u8,
    /// Power of two that rhs strides are aligned to
    pub rhs_stride_factor: u8,
    pub elem_lhs: StorageType,
    pub elem_rhs: StorageType,
    pub elem_out: StorageType,
    pub matrix_layout_lhs: MatrixBatchLayout,
    pub matrix_layout_rhs: MatrixBatchLayout,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulGlobalScale {
    Large,
    Medium,
    Small,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MatmulAutotuneAnalysis {
    pub scale_global: MatmulGlobalScale,
    pub kind: MatmulKind,
}

impl MatmulGlobalScale {
    pub fn from_size(m: usize, n: usize, k: usize) -> Self {
        if m < 512 && k < 512 && n < 512 {
            MatmulGlobalScale::Small
        } else if m < 2048 && k < 2048 && n < 2048 {
            MatmulGlobalScale::Medium
        } else {
            MatmulGlobalScale::Large
        }
    }
}

/// Whether it's a good idea to try and run double-buffered matmul.
pub fn should_tune_double_buffering(fused: bool, key: &MatmulAutotuneKey) -> bool {
    matches!(key.analysis.kind, MatmulKind::General)
        && match key.analysis.scale_global {
            MatmulGlobalScale::Large => true,
            MatmulGlobalScale::Medium => true,
            MatmulGlobalScale::Small => fused,
        }
}

impl MatmulAutotuneKey {
    /// Create the autotune key based on the shape of both lhs and rhs as well as the element type
    /// used for the calculation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<R: Runtime>(
        _client: &ComputeClient<R>,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        lhs_strides: &Strides,
        rhs_strides: &Strides,
        elem_lhs: StorageType,
        elem_rhs: StorageType,
        elem_out: StorageType,
        lhs_scheme: Option<&QuantScheme>,
        rhs_scheme: Option<&QuantScheme>,
    ) -> MatmulAutotuneKey {
        Self::from_parts(
            lhs_shape,
            rhs_shape,
            lhs_strides,
            rhs_strides,
            elem_lhs,
            elem_rhs,
            elem_out,
            lhs_scheme,
            rhs_scheme,
        )
    }

    /// [`MatmulAutotuneKey::generate`] without the (unused) client, so the key
    /// function stays testable without a runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        lhs_strides: &Strides,
        rhs_strides: &Strides,
        elem_lhs: StorageType,
        elem_rhs: StorageType,
        elem_out: StorageType,
        lhs_scheme: Option<&QuantScheme>,
        rhs_scheme: Option<&QuantScheme>,
    ) -> MatmulAutotuneKey {
        let ndims = lhs_shape.len();
        let m = lhs_shape[ndims - 2];
        let k = lhs_shape[ndims - 1];
        let n = rhs_shape[ndims - 1];

        let matrix_layout_lhs = matrix_batch_layout(lhs_strides, lhs_scheme);
        let matrix_layout_rhs = matrix_batch_layout(rhs_strides, rhs_scheme);

        let kind = MatmulKind::from(MatmulProblemSize {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        });

        // The alignment factors below are computed from the *anchored* dims —
        // the same bucketing the `m`/`n`/`k` fields get — never from the raw
        // values. A dimension carrying a runtime-dependent length (a KV-cache
        // width, a dynamic batch) would otherwise re-split every anchored
        // bucket into one key per pow2-alignment class of the raw value, and
        // the tuner would keep minting "new" problems the anchor deliberately
        // treats as equal. The trade-off matches the anchor's: the whole
        // bucket shares the kernel choice benchmarked on one representative,
        // and the launch still derives the legal line size from the real
        // tensors.
        let m_anchored = anchor(m, None, None, None);
        let n_anchored = anchor(n, None, None, None);
        let k_anchored = anchor(k, None, None, None);

        let lhs_pow2_factor = match matrix_layout_lhs {
            MatrixBatchLayout::Contiguous => pow2_factor(k_anchored),
            MatrixBatchLayout::MildlyPermuted { transposed, .. } => match transposed {
                true => pow2_factor(m_anchored),
                false => pow2_factor(k_anchored),
            },
            MatrixBatchLayout::HighlyPermuted => 0,
        };
        let rhs_pow2_factor = match matrix_layout_rhs {
            MatrixBatchLayout::Contiguous => pow2_factor(n_anchored),
            MatrixBatchLayout::MildlyPermuted { transposed, .. } => match transposed {
                true => pow2_factor(k_anchored),
                false => pow2_factor(n_anchored),
            },
            MatrixBatchLayout::HighlyPermuted => 0,
        };

        // The canonical tightest non-contiguous stride of each layout: the
        // row stride — `cols` for a contiguous `[.., rows, cols]`, `rows` for
        // its transposed view. Batch strides are products of these, so their
        // alignment can only be higher.
        let lhs_stride_factor = match matrix_layout_lhs {
            MatrixBatchLayout::Contiguous => stride_factor(k_anchored, elem_lhs),
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => stride_factor(m_anchored, elem_lhs),
            _ => 0,
        };
        let rhs_stride_factor = match matrix_layout_rhs {
            MatrixBatchLayout::Contiguous => stride_factor(n_anchored, elem_rhs),
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => stride_factor(k_anchored, elem_rhs),
            _ => 0,
        };

        let definition = MatmulProblemDefinition::new(
            m,
            n,
            k,
            lhs_pow2_factor,
            lhs_stride_factor,
            rhs_pow2_factor,
            rhs_stride_factor,
            elem_lhs,
            elem_rhs,
            elem_out,
            matrix_layout_lhs,
            matrix_layout_rhs,
        );
        let analysis = MatmulAutotuneAnalysis {
            scale_global: MatmulGlobalScale::from_size(m, n, k),
            kind,
        };

        Self::new(definition, analysis)
    }
}

/// Stride alignment (in powers of two of bytes) of the canonical row stride
/// `cols` — the tightest non-contiguous stride of the layout.
fn stride_factor(cols: usize, elem: StorageType) -> u8 {
    let bytes = (cols * elem.size_bits()) / 8;
    bytes.trailing_zeros().min(MAX_STRIDE_FACTOR) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: StorageType = StorageType::Scalar(ElemType::Float(FloatKind::F32));

    /// A contiguous `[batch, rows, cols]` problem pair, as the attention
    /// fallback produces: `scores[b, seq_q, kv] @ value[b, kv, head_dim]`.
    fn key(seq_q: usize, kv: usize, head_dim: usize) -> MatmulAutotuneKey {
        let lhs_shape = Shape::new([2, seq_q, kv]);
        let rhs_shape = Shape::new([2, kv, head_dim]);
        let lhs_strides = Strides::new(&[seq_q * kv, kv, 1]);
        let rhs_strides = Strides::new(&[kv * head_dim, head_dim, 1]);
        MatmulAutotuneKey::from_parts(
            &lhs_shape,
            &rhs_shape,
            &lhs_strides,
            &rhs_strides,
            F32,
            F32,
            F32,
            None,
            None,
        )
    }

    /// Every raw length inside one anchored bucket must produce the same key:
    /// a runtime-dependent dimension (a KV-cache width) may take any integer
    /// value, and re-splitting the bucket by the raw value's alignment retunes
    /// endlessly for problems the anchor treats as equal.
    #[test]
    fn raw_lengths_within_a_bucket_share_one_key() {
        // 65..=128 all anchor to the 128 bucket, at every alignment class:
        // odd, 2-, 4-, 8-, 16-aligned, and the exact power of two.
        let reference = key(64, 69, 64);
        for kv in [70, 76, 88, 96, 112, 128] {
            assert_eq!(reference, key(64, kv, 64), "kv {kv} split the bucket");
        }
    }

    /// Distinct anchored buckets still get distinct keys.
    #[test]
    fn buckets_stay_distinct() {
        assert_ne!(key(64, 128, 64), key(64, 200, 64));
        assert_ne!(key(64, 128, 64), key(1, 128, 64));
    }

    /// The transposed (`MildlyPermuted`) arm: a column-major LHS takes its alignment
    /// factors from `m` (the row count) rather than `k`, so raw `m` lengths inside one
    /// anchored bucket must share a key just as the contiguous `kv` case does. The
    /// [`key`] helper above is all-contiguous, so it never reaches this branch.
    #[test]
    fn transposed_lhs_shares_bucket_key() {
        // Column-major lhs `[b, m, k]`: `m` has stride 1, `k` has stride `m`, so
        // `row_stride < col_stride` and the layout is `MildlyPermuted { transposed }`.
        let key_t = |m: usize| {
            let (k, n) = (64usize, 64usize);
            MatmulAutotuneKey::from_parts(
                &Shape::new([2, m, k]),
                &Shape::new([2, k, n]),
                &Strides::new(&[k * m, 1, m]), // transposed lhs
                &Strides::new(&[k * n, n, 1]), // contiguous rhs
                F32,
                F32,
                F32,
                None,
                None,
            )
        };
        // 65..=128 all anchor to the 128 bucket, at every alignment class.
        let reference = key_t(69);
        for m in [70, 76, 88, 96, 112, 128] {
            assert_eq!(reference, key_t(m), "m {m} split the transposed bucket");
        }
    }
}
