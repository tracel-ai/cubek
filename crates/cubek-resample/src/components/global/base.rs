use crate::{
    components::{
        AccessPattern, GlobalOperation, GlobalReader, MemoryReader, MemoryReaderExpand, NdLayout,
        ScalarOperation, TapResult,
    },
    definition::{GlobalOperationKind, MemoryReaderKind},
};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[cube]
pub fn dispatch_reader<C: Numeric, P: AccessPattern>(
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
