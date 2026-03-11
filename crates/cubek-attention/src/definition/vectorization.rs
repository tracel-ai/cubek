use std::fmt::Debug;

use cubecl::{
    Runtime,
    client::ComputeClient,
    tensor_vector_size_parallel,
    zspace::{Shape, Strides},
};

use crate::definition::{AttentionGlobalTypes, AttentionIdent, AttentionProblem};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// Vector size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct AttentionVectorSizes {
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub mask: usize,
    pub out: usize,
}

impl AttentionVectorSizes {
    pub fn new_max<R: Runtime>(
        client: &ComputeClient<R>,
        global_dtypes: &AttentionGlobalTypes,
    ) -> Self {
        AttentionVectorSizes {
            query: client
                .io_optimized_vector_sizes(global_dtypes.query.size())
                .max()
                .unwrap(),
            key: client
                .io_optimized_vector_sizes(global_dtypes.key.size())
                .max()
                .unwrap(),
            value: client
                .io_optimized_vector_sizes(global_dtypes.value.size())
                .max()
                .unwrap(),
            // vectorized mask not always supported at the moment
            mask: 1,
            out: client
                .io_optimized_vector_sizes(global_dtypes.out.size())
                .max()
                .unwrap(),
        }
    }

    pub(crate) fn new_max_for_problem<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
    ) -> AttentionVectorSizes {
        AttentionVectorSizes {
            query: AttentionVectorSizes::find_vector_size(
                client,
                &problem.dims.shape(AttentionIdent::Query),
                problem.global_dtypes.query.size(),
            ),
            key: AttentionVectorSizes::find_vector_size(
                client,
                &problem.dims.shape(AttentionIdent::Key),
                problem.global_dtypes.key.size(),
            ),
            value: AttentionVectorSizes::find_vector_size(
                client,
                &problem.dims.shape(AttentionIdent::Value),
                problem.global_dtypes.value.size(),
            ),
            // vectorized mask not always supported at the moment
            mask: 1,
            out: AttentionVectorSizes::find_vector_size(
                client,
                &problem.dims.shape(AttentionIdent::Out),
                problem.global_dtypes.out.size(),
            ),
        }
    }

    fn find_vector_size<R: Runtime>(
        client: &ComputeClient<R>,
        shape: &[usize; 4],
        dtype_size: usize,
    ) -> usize {
        let supported_vector_sizes = client.io_optimized_vector_sizes(dtype_size);

        let n = shape.len();

        let row_major_strides = Strides::new(&{
            let mut strides = [0; 4];
            strides[n - 1] = 1;
            for i in (0..n - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        });
        let shape = Shape::new(*shape);

        tensor_vector_size_parallel(supported_vector_sizes, &shape, &row_major_strides, n - 1)
    }
}

impl From<&AttentionVectorSizes> for [usize; 5] {
    fn from(value: &AttentionVectorSizes) -> Self {
        [value.query, value.key, value.value, value.mask, value.out]
    }
}
