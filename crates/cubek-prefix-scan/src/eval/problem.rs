use crate::{definitions::PrefixScanKind, instructions::PrefixPrefixScanInstructionKind};
use cubecl::zspace::Shape;
use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixScanProblem {
    pub scan_kind: PrefixScanKind,
    pub scan_instruction_kind: PrefixPrefixScanInstructionKind,
    pub shape: Shape,
}

const SHAPES: &[&[usize]] = &[&[1_000_000_000]];

pub fn problems() -> Vec<CatalogEntry<PrefixScanProblem>> {
    let mut entries = Vec::new();

    for &shape in SHAPES {
        for &scan_instruction_kind in PrefixPrefixScanInstructionKind::ALL {
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
                    PrefixScanProblem {
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
