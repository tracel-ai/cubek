use crate::{
    components::{
        AccessPattern, GlobalOperation, GlobalReader, Layout, LayoutExpand, MemoryReader,
        MemoryReaderExpand, NdLayout, ReduceAxisPattern, ScalarOperation, TapResult,
    },
    definition::{AccessPatternKind, GlobalOperationKind, MemoryReaderKind},
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[cube(launch_unchecked)]
pub fn resample_kernel<C: Numeric>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    #[comptime] access_pattern_kind: AccessPatternKind,
    #[comptime] memory_reader_kind: MemoryReaderKind,
    #[comptime] operation_kind: GlobalOperationKind,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;

    if index >= output.len() {
        terminate!();
    }

    let out_coord = out_layout.from_linear(index);

    match access_pattern_kind {
        AccessPatternKind::ReduceAxisPattern(args) => {
            let access_pattern = ReduceAxisPattern {
                reduce_size: args.reduce_size,
            };

            dispatch_reader::<C, ReduceAxisPattern>(
                input,
                output,
                in_layout,
                index,
                out_coord,
                access_pattern,
                memory_reader_kind,
                operation_kind,
            );
        }
    }
}

#[cube]
fn dispatch_reader<C: Numeric, P: AccessPattern>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    in_layout: NdLayout,
    index: usize,
    out_coord: CoordsDyn,
    access_pattern: P,
    #[comptime] memory_reader_kind: MemoryReaderKind,
    #[comptime] operation_kind: GlobalOperationKind,
) {
    match memory_reader_kind {
        MemoryReaderKind::Global => {
            let reader = GlobalReader::<P>::init(out_coord, access_pattern);

            resample_inner::<C, P, GlobalReader<P>>(
                input,
                output,
                in_layout,
                index,
                reader,
                operation_kind,
            );
        }
    }
}

#[cube]
fn resample_inner<C: Numeric, P: AccessPattern, R: MemoryReader<P>>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    in_layout: NdLayout,
    index: usize,
    reader: R,
    #[comptime] operation_kind: GlobalOperationKind,
) {
    let mut accumulator = identity::<C>(operation_kind);
    let total_taps = reader.num_taps();
    for tap_idx in 0..total_taps {
        let tap = reader.read_at(input, &in_layout, tap_idx);

        let combined = combine::<C>(tap, operation_kind);

        accumulator = reduce::<C>(accumulator, combined, operation_kind);
    }
    let final_value = finalize::<C>(accumulator, operation_kind);
    output[index] = final_value;
}

#[cube]
fn identity<C: Numeric>(#[comptime] operation_kind: GlobalOperationKind) -> C {
    match operation_kind {
        GlobalOperationKind::Scalar(semiring) => ScalarOperation::<C>::identity(semiring),
    }
}

#[cube]
fn combine<C: Numeric>(tap: TapResult<C>, #[comptime] operation_kind: GlobalOperationKind) -> C {
    match operation_kind {
        GlobalOperationKind::Scalar(_) => ScalarOperation::<C>::combine(tap.value, tap.weight),
    }
}

#[cube]
fn reduce<C: Numeric>(
    accumulator: C,
    combined: C,
    #[comptime] operation_kind: GlobalOperationKind,
) -> C {
    match operation_kind {
        GlobalOperationKind::Scalar(semiring) => {
            ScalarOperation::<C>::reduce(accumulator, combined, semiring)
        }
    }
}

#[cube]
fn finalize<C: Numeric>(accumulator: C, #[comptime] operation_kind: GlobalOperationKind) -> C {
    match operation_kind {
        GlobalOperationKind::Scalar(_) => ScalarOperation::<C>::finalize(accumulator),
    }
}
