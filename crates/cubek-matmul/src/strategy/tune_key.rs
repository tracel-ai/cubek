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

/// Minimum byte-alignment factor required by async-copy and TMA readers.
const ASYNC_COPY_STRIDE_FACTOR: u32 = 4;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MatmulProblemDefinition {
    #[autotune(anchor)]
    pub m: usize,
    #[autotune(anchor)]
    pub n: usize,
    #[autotune(anchor)]
    pub k: usize,
    pub lhs_pow2_factor: u8,
    /// Async-copy alignment class for lhs strides: `4` when all relevant
    /// strides are 16-byte aligned, `0` otherwise.
    pub lhs_stride_factor: u8,
    pub rhs_pow2_factor: u8,
    /// Async-copy alignment class for rhs strides: `4` when all relevant
    /// strides are 16-byte aligned, `0` otherwise.
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

        // Vectorization factors are computed from the anchored dims so a
        // runtime-dependent length doesn't mint a key for every raw alignment
        // inside one shape bucket. Kernel launches still derive their legal
        // line sizes from the real tensors.
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

        // Async-copy and TMA candidates require every non-contiguous stride to
        // be 16-byte aligned. This legality bit must come from the real
        // strides: an anchored shape bucket can contain both aligned and
        // unaligned tensors, and sharing a cached async-copy winner across
        // those tensors would make the latter fail at launch. Collapsing the
        // value to two classes preserves bounded key cardinality.
        let lhs_stride_factor = match matrix_layout_lhs {
            MatrixBatchLayout::Contiguous => {
                async_copy_stride_factor(lhs_strides, ndims - 1, elem_lhs)
            }
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => async_copy_stride_factor(lhs_strides, ndims - 2, elem_lhs),
            _ => 0,
        };
        let rhs_stride_factor = match matrix_layout_rhs {
            MatrixBatchLayout::Contiguous => {
                async_copy_stride_factor(rhs_strides, ndims - 1, elem_rhs)
            }
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => async_copy_stride_factor(rhs_strides, ndims - 2, elem_rhs),
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

/// Classifies whether every non-contiguous stride meets the 16-byte alignment
/// requirement shared by async-copy and TMA readers.
fn async_copy_stride_factor(strides: &[usize], exclude_dim: usize, elem: StorageType) -> u8 {
    let factor = strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_dim)
        .map(|(_, stride)| (*stride * elem.size_bits()) / 8)
        .map(|bytes| bytes.trailing_zeros())
        .min()
        .unwrap_or(ASYNC_COPY_STRIDE_FACTOR);

    if factor >= ASYNC_COPY_STRIDE_FACTOR {
        ASYNC_COPY_STRIDE_FACTOR as u8
    } else {
        0
    }
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

    /// Raw lengths in one anchored bucket share a key when their async-copy
    /// legality is the same, but aligned and unaligned tensors never do.
    #[test]
    fn bucket_is_split_only_by_async_copy_legality() {
        let reference = key(64, 69, 64);
        for kv in [70, 74, 78] {
            assert_eq!(
                reference,
                key(64, kv, 64),
                "unaligned kv {kv} split the bucket"
            );
        }

        let aligned = key(64, 68, 64);
        for kv in [72, 76, 80, 96, 112, 128] {
            assert_eq!(aligned, key(64, kv, 64), "aligned kv {kv} split the bucket");
        }
        assert_ne!(reference, aligned);
    }

    /// Distinct anchored buckets still get distinct keys.
    #[test]
    fn buckets_stay_distinct() {
        assert_ne!(key(64, 128, 64), key(64, 200, 64));
        assert_ne!(key(64, 128, 64), key(1, 128, 64));
    }

    /// The transposed (`MildlyPermuted`) arm derives legality from the real
    /// column-major LHS strides as well.
    #[test]
    fn transposed_lhs_is_split_only_by_async_copy_legality() {
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
        let reference = key_t(69);
        for m in [70, 74, 78] {
            assert_eq!(reference, key_t(m), "unaligned m {m} split the bucket");
        }

        let aligned = key_t(68);
        for m in [72, 76, 80, 96, 112, 128] {
            assert_eq!(aligned, key_t(m), "aligned m {m} split the bucket");
        }
        assert_ne!(reference, aligned);
    }
}
