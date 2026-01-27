use std::fmt::Debug;

use cubecl::{Runtime, client::ComputeClient, tensor_line_size_parallel};

use crate::definition::{AttentionIdent, AttentionProblem, AttentionTileSize};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// Line size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct AttentionLineSizes {
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub mask: usize,
    pub out: usize,
}

impl AttentionLineSizes {
    pub(crate) fn new_max<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
    ) -> AttentionLineSizes {
        let find_line_size = |shape: &[usize; 4], dtype_size: usize| -> usize {
            let supported_line_sizes = client.io_optimized_line_sizes_unchecked(dtype_size);

            let n = shape.len();

            let row_major_strides = {
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            };

            tensor_line_size_parallel(supported_line_sizes, shape, &row_major_strides, n - 1)
        };

        AttentionLineSizes {
            query: find_line_size(
                &problem.dims.shape(AttentionIdent::Query),
                problem.global_dtypes.query.size(),
            ),
            key: find_line_size(
                &problem.dims.shape(AttentionIdent::Key),
                problem.global_dtypes.key.size(),
            ),
            value: find_line_size(
                &problem.dims.shape(AttentionIdent::Value),
                problem.global_dtypes.value.size(),
            ),
            // lined mask not always supported at the moment
            mask: 1,
            out: find_line_size(
                &problem.dims.shape(AttentionIdent::Out),
                problem.global_dtypes.out.size(),
            ),
        }
    }

    /// Cap line sizes to be compatible with the given tile size.
    /// Line sizes must evenly divide the corresponding tile dimensions.
    pub fn cap_to_tile_size(self, tile_size: &AttentionTileSize) -> AttentionLineSizes {
        fn cap(line_size: usize, tile_dim: u32) -> usize {
            let tile_dim = tile_dim as usize;
            if line_size > tile_dim || tile_dim % line_size != 0 {
                // Find largest power of 2 <= tile_dim that divides tile_dim
                let mut capped = 1;
                while capped * 2 <= tile_dim && tile_dim % (capped * 2) == 0 {
                    capped *= 2;
                }
                capped
            } else {
                line_size
            }
        }

        AttentionLineSizes {
            query: cap(self.query, tile_size.head_dim),
            key: cap(self.key, tile_size.head_dim),
            value: cap(self.value, tile_size.val_dim),
            mask: self.mask,
            out: cap(self.out, tile_size.val_dim),
        }
    }
}
