//! [`Stage`] / [`StageFamily`] / [`LoadStageFamily`] traits and the
//! [`StridedStageMemory`] impls that drive them, plus the
//! `ComptimeOption` / `Option` wrapper impls used by the optional-acc-stage
//! flow. The [`StridedStageMemory`] struct + inherent methods live in
//! `cubek_std::tile::variants::stage`; this module keeps the cubek-matmul-side
//! abstractions in one place.

use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{
    stage::StageMemoryConfig,
    tile::{SharedTile, StridedStage, Tile, TileScope, TilingLayout},
};

pub use cubek_std::stage::StridedStageMemory;

/// Stage that can be divided into tiles, with the same kind used by the tile
/// matmul readers.
#[cube]
pub trait Stage<ES: Numeric, IO: SliceVisibility = ReadOnly>:
    CubeType + Clone + Send + Sync + 'static
{
    /// Slices a tile with offset `(row, col)` from the stage and returns it.
    /// The [`TileScope`] generic lets the caller select the compute primitive
    /// that will consume this tile.
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, IO>;

    /// Wraps the whole stage as a [`TileKind::Stage`]-kind [`Tile`]. Used by
    /// the partition-matmul flow to drive
    /// [`Tile::mma_partition`](cubek_std::tile::Tile::mma_partition), which
    /// destructures the `Stage` kind to reach the underlying
    /// `StridedStage` payload.
    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc, IO>;
}

/// Stage family for any precision. The concrete `Stage` type is instantiated
/// with `(ES, NS, T)` — `NS` parameterizes the underlying allocation's vector
/// size; the produced `Stage<ES, IO>` no longer exposes it on its trait
/// surface.
pub trait StageFamily<IO: SliceVisibility = ReadOnly>: Send + Sync + 'static {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout>: Stage<ES, IO>;
}

/// Stage family that can be used as the target of a loader.
#[cube]
pub trait LoadStageFamily<IO: SliceVisibility = ReadOnly>: StageFamily {
    /// Create a new stage from the config and alignment.
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T>;
    /// Return the same stage with a different buffer index.
    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T>;
    /// Free the stage.
    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>);
}

pub struct StridedStageFamily;

impl StageFamily for StridedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = StridedStageMemory<ES, NS, T>;
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES, ReadOnly> for StridedStageMemory<ES, NS, T> {
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, ReadOnly> {
        let strided_tile = this.get_tile(tile);
        Tile::new_SharedTile(SharedTile::wrap::<NS>(strided_tile))
    }

    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc, ReadOnly> {
        Tile::new_Stage(StridedStage::wrap::<NS, T>(this))
    }
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES, ReadWrite>
    for StridedStageMemory<ES, NS, T>
{
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, ReadWrite> {
        let strided_tile = this.get_tile_mut(tile);
        Tile::new_SharedTile(SharedTile::wrap::<NS>(strided_tile))
    }

    fn as_stage_tile<Sc: TileScope>(_this: &Self) -> Tile<ES, Sc, ReadWrite> {
        panic!("StridedStageMemory: as_stage_tile is only supported for ReadOnly stages today")
    }
}

#[cube]
impl LoadStageFamily<ReadOnly> for StridedStageFamily {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        StridedStageMemory::new_aligned(alignment, config)
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.with_buffer_index(buffer_index)
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        unsafe { stage.free() };
    }
}

#[cube]
impl<ES: Numeric, IO: SliceVisibility, Inner: Stage<ES, IO>> Stage<ES, IO>
    for ComptimeOption<Inner>
{
    fn tile<Sc: TileScope>(this: &Self, tile: Coords2d) -> Tile<ES, Sc, IO> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::tile::<Sc>(inner, tile)
        } else {
            Tile::new_None()
        }
    }

    fn as_stage_tile<Sc: TileScope>(this: &Self) -> Tile<ES, Sc, IO> {
        #[comptime]
        if let ComptimeOption::Some(inner) = this {
            Inner::as_stage_tile::<Sc>(inner)
        } else {
            Tile::new_None()
        }
    }
}

#[cube]
impl<IO: SliceVisibility, S: LoadStageFamily<IO>> LoadStageFamily<IO> for Option<S> {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        ComptimeOption::new_Some(S::create(alignment, config))
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.as_ref().map(|s| S::with_buffer_index(s, index))
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        #[comptime]
        if let ComptimeOption::Some(inner) = stage {
            S::free(inner)
        }
    }
}

impl<IO: SliceVisibility, Inner: StageFamily<IO>> StageFamily<IO> for Option<Inner> {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = ComptimeOption<Inner::Stage<ES, NS, T>>;
}

pub use cubek_std::tile::PartitionBuffering;
