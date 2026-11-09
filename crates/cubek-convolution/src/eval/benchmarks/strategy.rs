use cubek_test_utils::CatalogEntry;

use crate::{
    AcceleratedTileKind, ConvAlgorithm, Strategy,
    kernels::tiled::{ConvRoutineKind, ConvSettings, CubeStagedConvRoutine},
};

/// The settings pinned for each instruction, as a bench instrument beside the tuned row: the
/// lead, one tile of rows, the narrow instruction, and a grid held whole over a map of its width.
const CMMA: [ConvSettings; 5] = [
    CubeStagedConvRoutine::F2X2_P2X2,
    CubeStagedConvRoutine::F1X2_P1X2,
    CubeStagedConvRoutine::F1X1_P1X2_T8,
    CubeStagedConvRoutine::F7X1_P1X2_T8,
    CubeStagedConvRoutine::F1X1_P1X1_T8_A7,
];

/// The settings pinned through the manual mma: the lead at both widths, one tile of rows, and a
/// grid of 32 rows at one stage and at two in flight.
const MMA: [ConvSettings; 5] = [
    CubeStagedConvRoutine::F2X2_P2X2.mma(),
    CubeStagedConvRoutine::F2X2_P2X2.mma().width_one(),
    CubeStagedConvRoutine::F1X2_P1X2.mma(),
    CubeStagedConvRoutine::F1X2_P2X2.mma(),
    CubeStagedConvRoutine::F1X2_P2X2.mma().double_ring(),
];

/// A row of the cube staged routine, named by its strategy: the routine and, where pinned, its
/// settings.
fn cube_staged(settings: Option<ConvSettings>) -> CatalogEntry<Strategy> {
    let strategy = Strategy::Tile {
        routine: ConvRoutineKind::CubeStaged,
        settings,
    };
    let label = match settings {
        Some(settings) => format!("Cube staged / {settings}"),
        None => "Cube staged, tuned over the device's settings".to_string(),
    };
    CatalogEntry::new(strategy.to_string(), label, strategy)
}

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    let mut strategies = vec![
        CatalogEntry::new(
            "simple_sync_cyclic_cmma",
            "SimpleSyncCyclic / Cmma (inferred)",
            Strategy::Inferred {
                algorithm: ConvAlgorithm::SimpleSyncCyclic,
                tile_kind: AcceleratedTileKind::Cmma,
            },
        ),
        CatalogEntry::new(
            "simple_sync_cyclic_mma",
            "SimpleSyncCyclic / Mma (inferred)",
            Strategy::Inferred {
                algorithm: ConvAlgorithm::SimpleSyncCyclic,
                tile_kind: AcceleratedTileKind::Mma,
            },
        ),
        cube_staged(None),
    ];
    strategies.extend(
        CMMA.into_iter()
            .chain(MMA)
            .map(|settings| cube_staged(Some(settings))),
    );
    strategies
}
