use cubek_test_utils::CatalogEntry;

use crate::{
    AcceleratedTileKind, ConvAlgorithm, Strategy,
    kernels::tiled::{ConvRoutineKind, ConvSettings, CubeStagedConvRoutine},
};

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    vec![
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
        CatalogEntry::new(
            "cube_staged",
            "Cube staged, first setting that serves",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: None,
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_grid_2x2",
            "Cube staged / Cmma, lines, 2x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_2X2),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_grid_1x2",
            "Cube staged / Cmma, lines, 1x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_1X2),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_8x32",
            "Cube staged / Cmma, lines, 8x32",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_8X32),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2_width_1",
            "Cube staged / Mma, width 1, 2x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_2X2.mma().width_one()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2",
            "Cube staged / Mma, lines, 2x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_2X2.mma()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2",
            "Cube staged / Mma, lines, 1x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::GRID_1X2.mma()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2_whole_walk",
            "Cube staged / Mma, lines, 2x2 grid, 7 steps a stage",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(ConvSettings {
                    stage_steps: 7,
                    ..CubeStagedConvRoutine::GRID_2X2.mma()
                }),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2_whole_walk",
            "Cube staged / Mma, lines, 1x2 grid, 7 steps a stage",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(ConvSettings {
                    stage_steps: 7,
                    ..CubeStagedConvRoutine::GRID_1X2.mma()
                }),
            },
        ),
    ]
}
