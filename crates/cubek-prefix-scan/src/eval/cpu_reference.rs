use crate::{
    ScanKind,
    instructions::{Maximum, Minimum, Product, ScanInstruction, ScanInstructionKind, Sum},
};
use cubek_test_utils::HostData;

pub fn cpu_reference_prefix_scan_generic_instruction<I: ScanInstruction>(
    input: &HostData,
    scan_kind: ScanKind,
) -> Vec<f32> {
    match scan_kind {
        ScanKind::Exclusive => (0..input.shape[0])
            .scan(I::identity::<f32>(), |accumulator, index| {
                let prev = *accumulator;
                *accumulator = I::combine::<f32>(*accumulator, input.get_f32(&[index]));
                Some(prev)
            })
            .collect::<Vec<_>>(),
        ScanKind::Inclusive => (0..input.shape[0])
            .scan(I::identity::<f32>(), |accumulator, index| {
                *accumulator = I::combine::<f32>(*accumulator, input.get_f32(&[index]));
                Some(*accumulator)
            })
            .collect::<Vec<_>>(),
    }
}

pub fn cpu_reference_prefix_scan(
    input: &HostData,
    scan_kind: ScanKind,
    scan_instruction_kind: ScanInstructionKind,
) -> Vec<f32> {
    match scan_instruction_kind {
        ScanInstructionKind::Sum => {
            cpu_reference_prefix_scan_generic_instruction::<Sum>(input, scan_kind)
        }
        ScanInstructionKind::Product => {
            cpu_reference_prefix_scan_generic_instruction::<Product>(input, scan_kind)
        }
        ScanInstructionKind::Minimum => {
            cpu_reference_prefix_scan_generic_instruction::<Minimum>(input, scan_kind)
        }
        ScanInstructionKind::Maximum => {
            cpu_reference_prefix_scan_generic_instruction::<Maximum>(input, scan_kind)
        }
    }
}
