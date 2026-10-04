use crate::{
    definitions::PrefixScanKind,
    instructions::{
        Maximum, Minimum, PrefixPrefixScanInstructionKind, PrefixScanInstruction, Product, Sum,
    },
};
use cubek_test_utils::HostData;

pub fn cpu_reference_prefix_scan_generic_instruction<I: PrefixScanInstruction>(
    input: &HostData,
    scan_kind: PrefixScanKind,
) -> Vec<f32> {
    match scan_kind {
        PrefixScanKind::Exclusive => (0..input.shape[0])
            .scan(I::identity::<f32>(), |accumulator, index| {
                let prev = *accumulator;
                *accumulator = I::combine::<f32>(*accumulator, input.get_f32(&[index]));
                Some(prev)
            })
            .collect::<Vec<_>>(),
        PrefixScanKind::Inclusive => (0..input.shape[0])
            .scan(I::identity::<f32>(), |accumulator, index| {
                *accumulator = I::combine::<f32>(*accumulator, input.get_f32(&[index]));
                Some(*accumulator)
            })
            .collect::<Vec<_>>(),
    }
}

pub fn cpu_reference_prefix_scan(
    input: &HostData,
    scan_kind: PrefixScanKind,
    scan_instruction_kind: PrefixPrefixScanInstructionKind,
) -> Vec<f32> {
    match scan_instruction_kind {
        PrefixPrefixScanInstructionKind::Sum => {
            cpu_reference_prefix_scan_generic_instruction::<Sum>(input, scan_kind)
        }
        PrefixPrefixScanInstructionKind::Product => {
            cpu_reference_prefix_scan_generic_instruction::<Product>(input, scan_kind)
        }
        PrefixPrefixScanInstructionKind::Minimum => {
            cpu_reference_prefix_scan_generic_instruction::<Minimum>(input, scan_kind)
        }
        PrefixPrefixScanInstructionKind::Maximum => {
            cpu_reference_prefix_scan_generic_instruction::<Maximum>(input, scan_kind)
        }
    }
}
