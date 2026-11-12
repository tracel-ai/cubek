use crate::{
    base::PrefixScanKind, eval::tune::PrefixScanTuneKey, instructions::PrefixScanInstructionKind,
};
use cubecl::frontend::Scalar;
use cubek_test_utils::CatalogEntry;

const SHAPES: &[&[usize]] = &[&[1_000_000_000]];

pub fn problems() -> Vec<CatalogEntry<PrefixScanTuneKey>> {
    let mut entries = Vec::new();

    for &shape in SHAPES {
        for &scan_instruction_kind in PrefixScanInstructionKind::ALL {
            for &scan_kind in PrefixScanKind::ALL {
                entries.push(CatalogEntry::new(
                    format!(
                        "{}_{}_{}",
                        scan_kind.label().to_lowercase(),
                        scan_instruction_kind.label().to_lowercase(),
                        shape[0],
                    ),
                    format!(
                        "{} {} over {} elements",
                        scan_kind.label(),
                        scan_instruction_kind.label(),
                        shape[0],
                    ),
                    PrefixScanTuneKey {
                        dtype: f32::elem_type_native(),
                        scan_kind,
                        scan_instruction_kind,
                        shape: shape.into(),
                    },
                ));
            }
        }
    }

    entries
}
