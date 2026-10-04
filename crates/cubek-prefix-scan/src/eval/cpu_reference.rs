use crate::{
    base::PrefixScanKind,
    instructions::{
        Maximum, Minimum, PrefixScanInstruction, PrefixScanInstructionKind, Product, Sum,
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
    scan_instruction_kind: PrefixScanInstructionKind,
) -> Vec<f32> {
    match scan_instruction_kind {
        PrefixScanInstructionKind::Sum => {
            cpu_reference_prefix_scan_generic_instruction::<Sum>(input, scan_kind)
        }
        PrefixScanInstructionKind::Product => {
            cpu_reference_prefix_scan_generic_instruction::<Product>(input, scan_kind)
        }
        PrefixScanInstructionKind::Minimum => {
            cpu_reference_prefix_scan_generic_instruction::<Minimum>(input, scan_kind)
        }
        PrefixScanInstructionKind::Maximum => {
            cpu_reference_prefix_scan_generic_instruction::<Maximum>(input, scan_kind)
        }
    }
}
