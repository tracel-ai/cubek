use std::fmt::Debug;

use cubecl::{Runtime, client::ComputeClient, tensor_line_size_parallel};
use cubek_std::test_utils::StrideSpec;

use crate::launch::{AttentionDefinition, AttentionIdent};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// Line size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct AttentionLineSizes {
    pub query: u8,
    pub key: u8,
    pub value: u8,
    pub mask: u8,
    pub out: u8,
}

impl AttentionLineSizes {
    pub(crate) fn new_max<R: Runtime>(
        client: &ComputeClient<R>,
        definition: &AttentionDefinition,
    ) -> AttentionLineSizes {
        let find_line_size = |shape: &[usize; 4], dtype_size: usize| -> u8 {
            let supported_line_sizes = client.io_optimized_line_sizes_unchecked(dtype_size);

            tensor_line_size_parallel(
                supported_line_sizes,
                shape,
                &StrideSpec::RowMajor.compute_strides(shape),
                shape.len() - 1,
            )
        };

        AttentionLineSizes {
            query: find_line_size(
                &definition.dims.shape(AttentionIdent::Query),
                definition.global_dtypes.query.size(),
            ),
            key: find_line_size(
                &definition.dims.shape(AttentionIdent::Key),
                definition.global_dtypes.key.size(),
            ),
            value: find_line_size(
                &definition.dims.shape(AttentionIdent::Value),
                definition.global_dtypes.value.size(),
            ),
            // lined mask not always supported at the moment
            mask: 1,
            out: find_line_size(
                &definition.dims.shape(AttentionIdent::Out),
                definition.global_dtypes.out.size(),
            ),
        }
    }
}
