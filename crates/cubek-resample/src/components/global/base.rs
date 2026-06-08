use crate::{
    components::{
        GlobalOperation as GlobalOperationTrait, GlobalReader, Layout, LayoutExpand, MemoryReader,
        MemoryReaderExpand, NdLayout, ReduceAxisPattern, ScalarOperation,
    },
    definition::GlobalOperation,
};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn resample_kernel<C: Numeric>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    #[comptime] global_operation: GlobalOperation,
    #[define(C)] _dtype: StorageType,
) {
    let linear_idx = ABSOLUTE_POS as usize;
    if linear_idx >= output.len() {
        terminate!();
    }

    let out_coord = out_layout.from_linear(linear_idx);

    let access_pattern = ReduceAxisPattern {
        reduce_size: 1 as u32,
    };

    let reader = GlobalReader::<ReduceAxisPattern>::init(out_coord, access_pattern);

    let mut accumulator = identity::<C>(global_operation);

    let num_taps = reader.num_taps();
    for tap_idx in 0..num_taps {
        let tap = reader.read_at(input, &in_layout, tap_idx);

        let combined = ScalarOperation::<C>::combine(tap.value, tap.weight);

        accumulator = reduce::<C>(accumulator, combined, global_operation);
    }

    let final_val = ScalarOperation::<C>::finalize(accumulator);

    output[linear_idx] = final_val;
}

#[cube]
fn identity<C: Numeric>(#[comptime] global_operation: GlobalOperation) -> C {
    match global_operation {
        GlobalOperation::Scalar(semiring) => ScalarOperation::<C>::identity(semiring),
    }
}

#[cube]
fn reduce<C: Numeric>(
    accumulator: C,
    combined: C,
    #[comptime] global_operation: GlobalOperation,
) -> C {
    match global_operation {
        GlobalOperation::Scalar(semiring) => {
            ScalarOperation::<C>::reduce(accumulator, combined, semiring)
        }
    }
}
