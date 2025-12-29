use std::fmt::Debug;

use cubecl::{Runtime, client::ComputeClient, tensor_line_size_parallel};

use crate::definition::{AttentionIdent, AttentionProblem};

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
}
