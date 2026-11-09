use cubek_test_utils::CatalogEntry;

use crate::{
    AcceleratedTileKind, ConvAlgorithm, Strategy,
    kernels::tiled::{ConvRoutineKind, CubeStagedConvRoutine},
};

/// Planes across the output rows a cube holds in the rows that split them.
const ACROSS: [usize; 3] = [2, 4, 7];

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    let across = [
        (
            "cmma_8x32",
            "Cmma, lines, 8x32",
            CubeStagedConvRoutine::F1X1_P1X2_T8,
        ),
        (
            "cmma_grid_1x2",
            "Cmma, lines, 1x2 grid",
            CubeStagedConvRoutine::F1X2_P1X2,
        ),
        (
            "mma_grid_1x2",
            "Mma, lines, 1x2 grid",
            CubeStagedConvRoutine::F1X2_P1X2.mma(),
        ),
    ]
    .into_iter()
    .flat_map(|(id, label, settings)| {
        ACROSS.map(|planes| {
            CatalogEntry::new(
                format!("cube_staged_{id}_across_{planes}"),
                format!("Cube staged / {label}, {planes} planes across"),
                Strategy::Tile {
                    routine: ConvRoutineKind::CubeStaged,
                    settings: Some(settings.across(planes)),
                },
            )
        })
    });
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
                settings: Some(CubeStagedConvRoutine::F2X2_P2X2),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_grid_1x2",
            "Cube staged / Cmma, lines, 1x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X2_P1X2),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_8x32",
            "Cube staged / Cmma, lines, 8x32",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X1_P1X2_T8),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2_width_1",
            "Cube staged / Mma, width 1, 2x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F2X2_P2X2.mma().width_one()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2",
            "Cube staged / Mma, lines, 2x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F2X2_P2X2.mma()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2",
            "Cube staged / Mma, lines, 1x2 grid",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X2_P1X2.mma()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_2x2_whole_walk",
            "Cube staged / Mma, lines, 2x2 grid, 7 steps a stage",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F2X2_P2X2.mma().stage_steps(7)),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2_whole_walk",
            "Cube staged / Mma, lines, 1x2 grid, 7 steps a stage",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X2_P1X2.mma().stage_steps(7)),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_grid_2x2_on_1x2",
            "Cube staged / Cmma, lines, 2x2 on 1x2 planes",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F2X2_P1X2),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2_on_2x2",
            "Cube staged / Mma, lines, 1x2 on 2x2 planes",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X2_P2X2.mma()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_mma_grid_1x2_on_2x2_double",
            "Cube staged / Mma, lines, 1x2 on 2x2 planes, two stages in flight",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X2_P2X2.mma().double_ring()),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_56_rows",
            "Cube staged / Cmma, lines, 56 rows of 8x32",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F7X1_P1X2_T8),
            },
        ),
        CatalogEntry::new(
            "cube_staged_cmma_8x32_across_7_one_plane",
            "Cube staged / Cmma, lines, 8x32, 7 rows on one plane each",
            Strategy::Tile {
                routine: ConvRoutineKind::CubeStaged,
                settings: Some(CubeStagedConvRoutine::F1X1_P1X1_T8_A7),
            },
        ),
    ];
    strategies.extend(across);
    strategies
}
