use cubek_test_utils::CatalogEntry;

use crate::{
    AcceleratedTileKind, ConvAlgorithm, Strategy, UnitStrategy, UnitTiling,
    kernels::tiled::{ConvRoutineKind, ConvSettings, CubeStagedConvRoutine},
};

/// The settings pinned for each cmma instruction, beside the tuned row: the base setting, two
/// tiles of rows over two planes, one tile of rows, the narrow instruction, and a grid held whole
/// over a map of its width. The `T8` rows refuse where the device lacks the 8x32 instruction.
const CMMA: [ConvSettings; 6] = [
    CubeStagedConvRoutine::F2X2_P2X2_T16,
    CubeStagedConvRoutine::F2X2_P1X2_T16,
    CubeStagedConvRoutine::F1X2_P1X2_T16,
    CubeStagedConvRoutine::F1X1_P1X2_T8,
    CubeStagedConvRoutine::F7X1_P1X2_T8,
    CubeStagedConvRoutine::F1X1_P1X1_T8_A7,
];

/// The settings pinned through the manual mma: the base setting at both widths, one tile of rows,
/// and a grid of 32 rows at one stage and at two in flight.
const MMA: [ConvSettings; 5] = [
    CubeStagedConvRoutine::F2X2_P2X2_T16.mma(),
    CubeStagedConvRoutine::F2X2_P2X2_T16.mma().width_one(),
    CubeStagedConvRoutine::F1X2_P1X2_T16.mma(),
    CubeStagedConvRoutine::F1X2_P2X2_T16.mma(),
    CubeStagedConvRoutine::F1X2_P2X2_T16.mma().double_ring(),
];

/// A row of the cube staged routine, named by its strategy: the routine and, where pinned, its
/// settings. Without settings it is the tuned launch, which floors on the unit routine.
fn cube_staged(settings: Option<ConvSettings>) -> CatalogEntry<Strategy> {
    let routine = match settings {
        Some(_) => ConvRoutineKind::CubeStaged,
        None => ConvRoutineKind::Tuned,
    };
    let strategy = Strategy::Tile { routine, settings };
    let label = match settings {
        Some(settings) => format!("Cube staged / {settings}"),
        None => "Tuned: cube staged settings the device offers, the unit routine where none serves"
            .to_string(),
    };
    CatalogEntry::new(strategy.to_string(), label, strategy)
}

/// The unit routine of the tile convolution at the settings it derives from the device.
fn unit_registers() -> CatalogEntry<Strategy> {
    let strategy = Strategy::Tile {
        routine: ConvRoutineKind::UnitRegisters,
        settings: None,
    };
    CatalogEntry::new(
        strategy.to_string(),
        "Unit registers, at the device's settings",
        strategy,
    )
}

/// The register-tiled `f32` convolution at its balanced tiling, which a device with no matrix
/// instruction runs.
fn unit_balanced() -> CatalogEntry<Strategy> {
    let strategy = Strategy::Unit(UnitStrategy::Fixed(UnitTiling::BALANCED));
    CatalogEntry::new(strategy.to_string(), "Unit / balanced tiling", strategy)
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
        unit_registers(),
        unit_balanced(),
    ];
    strategies.extend(
        CMMA.into_iter()
            .chain(MMA)
            .map(|settings| cube_staged(Some(settings))),
    );
    strategies
}
