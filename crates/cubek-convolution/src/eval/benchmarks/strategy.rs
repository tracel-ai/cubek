use cubek_test_utils::CatalogEntry;

use super::settings;
use crate::{
    AcceleratedTileKind, ConvAlgorithm, Strategy, UnitStrategy, UnitTiling,
    kernels::tiled::{ConvRoutineKind, CubeStagedConvSettings},
};

const PLANE_WINDOW_LABEL: &str = "Plane window, at the device's settings";
const UNIT_BALANCED_LABEL: &str = "Unit / balanced tiling";
const FIRST_SERVED_LABEL: &str = "Cube staged / first listed setting that serves";

/// The settings pinned, beside the row at the first setting that serves: the base setting at both
/// widths, two tiles of rows over two planes, one tile of rows, the narrow instruction, and a grid
/// held whole over a map of its width. The `T8` rows refuse where the tile picks no instruction of
/// eight rows.
const PINNED: [CubeStagedConvSettings; 8] = [
    settings::F2X2_P2X2_T16,
    settings::F2X2_P2X2_T16.width_one(),
    settings::F2X2_P1X2_T16,
    settings::F1X2_P1X2_T16,
    settings::F1X2_P2X2_T16,
    settings::F1X1_P1X2_T8,
    settings::F7X1_P1X2_T8,
    settings::F1X1_P1X1_T8_A7,
];

/// A row of the cube staged routine, named by its strategy: the routine and, where pinned, its
/// settings. Without settings it runs the first setting the device lists that serves.
fn cube_staged(settings: Option<CubeStagedConvSettings>) -> CatalogEntry<Strategy> {
    let label = match settings {
        Some(settings) => format!("Cube staged / {settings}"),
        None => FIRST_SERVED_LABEL.to_string(),
    };
    let strategy = Strategy::Tile(ConvRoutineKind::CubeStaged(settings));
    CatalogEntry::new(strategy.to_string(), label, strategy)
}

/// The plane window routine of the tile convolution at the settings it derives from the device.
fn plane_window() -> CatalogEntry<Strategy> {
    let strategy = Strategy::Tile(ConvRoutineKind::PlaneWindow);
    CatalogEntry::new(strategy.to_string(), PLANE_WINDOW_LABEL, strategy)
}

/// The register-tiled `f32` convolution at its balanced tiling, which a device with no matrix
/// instruction runs.
fn unit_balanced() -> CatalogEntry<Strategy> {
    let strategy = Strategy::Unit(UnitStrategy::Fixed(UnitTiling::BALANCED));
    CatalogEntry::new(strategy.to_string(), UNIT_BALANCED_LABEL, strategy)
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
        plane_window(),
        unit_balanced(),
    ];
    strategies.extend(
        PINNED
            .into_iter()
            .map(|settings| cube_staged(Some(settings))),
    );
    strategies
}
