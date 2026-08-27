use cubecl::{
    AutotuneKey, Runtime,
    client::ComputeClient,
    ir::ElemType,
    quant::scheme::QuantScheme,
    tune::anchor,
    zspace::{Shape, Strides},
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
    pub elem_lhs: ElemType,
    pub elem_rhs: ElemType,
    pub elem_out: ElemType,
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
        elem_lhs: ElemType,
        elem_rhs: ElemType,
        elem_out: ElemType,
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
        elem_lhs: ElemType,
        elem_rhs: ElemType,
        elem_out: ElemType,
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

        // The vectorization factors below are computed from the *anchored* dims,
        // the same bucketing the `m`/`n`/`k` fields get, never from the raw
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

        let lhs_stride_factor = stride_factor(lhs_strides, &matrix_layout_lhs, elem_lhs);
        let rhs_stride_factor = stride_factor(rhs_strides, &matrix_layout_rhs, elem_rhs);

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
            // From the anchored dims, like every derived field above: the raw
            // thresholds (512/2048) otherwise split an anchored bucket at its
            // maximum: raw 257..511 keyed `Small` while exactly 512 keyed
            // `Medium`, so a runtime-dependent dim landing on the bucket
            // ceiling minted a "new" problem and re-tuned mid-run.
            scale_global: MatmulGlobalScale::from_size(m_anchored, n_anchored, k_anchored),
            kind,
        };

        Self::new(definition, analysis)
    }
}

/// Minimum non-contiguous stride alignment in powers of two of bytes.
fn stride_factor(strides: &Strides, layout: &MatrixBatchLayout, elem: ElemType) -> u8 {
    let exclude_dim = match layout {
        MatrixBatchLayout::Contiguous => strides.len() - 1,
        MatrixBatchLayout::MildlyPermuted {
            transposed: true,
            batch_swap: false,
        } => strides.len() - 2,
        _ => return 0,
    };

    strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_dim)
        .map(|(_, stride)| (stride * elem.size_bits()) / 8)
        .map(|bytes| bytes.trailing_zeros())
        .min()
        .unwrap_or(MAX_STRIDE_FACTOR)
        .min(MAX_STRIDE_FACTOR) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::{MatrixLayout, launch::tma::stride_align_bits};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

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

    /// Shapes in one anchored bucket still need distinct keys when their actual stride alignment
    /// changes, because that determines whether async-copy kernels are legal.
    #[test]
    fn stride_alignment_splits_an_anchor_bucket() {
        assert_eq!(key(64, 69, 64), key(64, 71, 64));
        assert_ne!(key(64, 69, 64), key(64, 70, 64));
        assert_ne!(key(64, 70, 64), key(64, 76, 64));
    }

    /// The bucket maximum shares its anchored dimensions and scale. Stride alignment may still
    /// split the full key because it changes kernel legality.
    #[test]
    fn bucket_maxima_share_anchored_dimensions_and_scale() {
        let reference = key(64, 460, 64);
        let maximum = key(64, 512, 64);
        assert_eq!(reference.definition.k, maximum.definition.k);
        assert_eq!(reference.analysis, maximum.analysis);
        let reference = key(64, 1500, 64);
        let maximum = key(64, 2048, 64);
        assert_eq!(reference.definition.k, maximum.definition.k);
        assert_eq!(reference.analysis, maximum.analysis);
    }

    /// Distinct anchored buckets still get distinct keys.
    #[test]
    fn buckets_stay_distinct() {
        assert_ne!(key(64, 512, 64), key(64, 513, 64));
        assert_ne!(key(64, 128, 64), key(64, 200, 64));
        assert_ne!(key(64, 128, 64), key(1, 128, 64));
    }

    /// The transposed (`MildlyPermuted`) arm must likewise use the actual column stride. The
    /// [`key`] helper above is all-contiguous, so it never reaches this branch.
    #[test]
    fn transposed_lhs_stride_alignment_splits_an_anchor_bucket() {
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
        assert_eq!(key_t(69), key_t(71));
        assert_ne!(key_t(69), key_t(70));
        assert_ne!(key_t(70), key_t(76));
    }

    #[test]
    fn async_copy_legality_must_distinguish_padded_row_stride() {
        let lhs_shape = Shape::new([65_536, 128]);
        let rhs_shape = Shape::new([128, 2]);
        let rhs_strides = Strides::new(&[1, 128]);
        let aligned_strides = Strides::new(&[128, 1]);
        let padded_strides = Strides::new(&[130, 1]);

        let key = |lhs_strides: &Strides| {
            MatmulAutotuneKey::from_parts(
                &lhs_shape,
                &rhs_shape,
                lhs_strides,
                &rhs_strides,
                F32,
                F32,
                F32,
                None,
                None,
            )
        };

        assert!(stride_align_bits(&aligned_strides, &MatrixLayout::RowMajor, &F32) >= 4);
        assert!(stride_align_bits(&padded_strides, &MatrixLayout::RowMajor, &F32) < 4);
        assert_ne!(key(&aligned_strides), key(&padded_strides));
    }

    #[test]
    fn rhs_async_copy_legality_distinguishes_padded_row_stride() {
        let lhs_shape = Shape::new([2, 128]);
        let rhs_shape = Shape::new([128, 128]);
        let lhs_strides = Strides::new(&[128, 1]);
        let aligned_strides = Strides::new(&[128, 1]);
        let padded_strides = Strides::new(&[130, 1]);

        let key = |rhs_strides: &Strides| {
            MatmulAutotuneKey::from_parts(
                &lhs_shape,
                &rhs_shape,
                &lhs_strides,
                rhs_strides,
                F32,
                F32,
                F32,
                None,
                None,
            )
        };

        assert!(stride_align_bits(&aligned_strides, &MatrixLayout::RowMajor, &F32) >= 4);
        assert!(stride_align_bits(&padded_strides, &MatrixLayout::RowMajor, &F32) < 4);
        assert_ne!(key(&aligned_strides), key(&padded_strides));
    }

    #[test]
    fn transposed_rhs_async_copy_legality_distinguishes_padded_column_stride() {
        let lhs_shape = Shape::new([2, 128]);
        let rhs_shape = Shape::new([128, 128]);
        let lhs_strides = Strides::new(&[128, 1]);
        let aligned_strides = Strides::new(&[1, 128]);
        let padded_strides = Strides::new(&[1, 130]);

        let key = |rhs_strides: &Strides| {
            MatmulAutotuneKey::from_parts(
                &lhs_shape,
                &rhs_shape,
                &lhs_strides,
                rhs_strides,
                F32,
                F32,
                F32,
                None,
                None,
            )
        };

        assert!(stride_align_bits(&aligned_strides, &MatrixLayout::ColMajor, &F32) >= 4);
        assert!(stride_align_bits(&padded_strides, &MatrixLayout::ColMajor, &F32) < 4);
        assert_ne!(key(&aligned_strides), key(&padded_strides));
    }

    #[test]
    fn batch_stride_contributes_to_async_copy_legality() {
        let lhs_shape = Shape::new([2, 128, 128]);
        let rhs_shape = Shape::new([2, 128, 2]);
        let rhs_strides = Strides::new(&[256, 2, 1]);
        let aligned_strides = Strides::new(&[16_384, 128, 1]);
        let padded_strides = Strides::new(&[16_386, 128, 1]);

        let key = |lhs_strides: &Strides| {
            MatmulAutotuneKey::from_parts(
                &lhs_shape,
                &rhs_shape,
                lhs_strides,
                &rhs_strides,
                F32,
                F32,
                F32,
                None,
                None,
            )
        };

        let aligned = key(&aligned_strides);
        let padded = key(&padded_strides);
        assert_eq!(aligned.definition.lhs_stride_factor, 9);
        assert_eq!(padded.definition.lhs_stride_factor, 3);
        assert_eq!(
            padded.definition.lhs_stride_factor as u32,
            stride_align_bits(&padded_strides, &MatrixLayout::RowMajor, &F32)
        );
        assert_ne!(aligned, padded);
    }
}
