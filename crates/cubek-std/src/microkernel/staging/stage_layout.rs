use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Space, StageStorage, stage::RowChunks};

/// What a fragment that reads a tile off a pointer and a stride declines: swizzled tiles.
pub const SWIZZLED_STAGE_UNREAD: Refusal = Refusal::Declined {
    why: "a cmma fragment reads a stage tile by a stride, which swizzled tiles do not keep",
};

/// How a stage's rows lie over the shared memory banks.
///
/// Where a row spans a whole number of the banks' 128 bytes, as a head dim or a step of `K` of 64
/// or 128 in `f16` does, a matrix instruction, loading the same columns of several rows at once,
/// reads one bank several times over. Padding each row by a chunk starts the next row on other
/// banks, at a chunk a row; blocking the stage into the instruction's tiles packs one tile's rows
/// into consecutive bytes instead, at none, and swizzling a tile's chunks spreads its rows over the
/// banks in place, also at none. Which reads faster is the device's to say, so each is a setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageLayout {
    /// Row-major over the whole stage, the rows in order.
    Strided,
    /// Row-major over the whole stage, each row one chunk longer.
    PaddedRows,
    /// In tiles of the instruction, one after another, their rows in order.
    Blocks,
    /// In tiles of the instruction, each of their rows one chunk longer.
    PaddedBlocks,
    /// In tiles of the instruction, each row's chunks swizzled: what a fragment reads that hands
    /// every row its own address, as the tile DSL's own transport does; a cmma fragment reads a
    /// tile off a pointer and a stride, and cannot.
    SwizzledBlocks,
}

impl StageLayout {
    /// The storage a stage over `stage` is held in, the instruction's tile `tile` cells along each
    /// of its axes, which are the stage's axes in the order the stage lays them.
    pub fn storage(self, stage: &Space, tile: Vec<(Axis, usize)>) -> StageStorage {
        let (block, chunks) = match self {
            Self::Strided => return StageStorage::Strided,
            Self::PaddedRows => {
                let whole = tile
                    .into_iter()
                    .map(|(axis, _)| (axis, stage.extent(axis)))
                    .collect();
                (whole, RowChunks::Padded)
            }
            Self::Blocks => (tile, RowChunks::InOrder),
            Self::PaddedBlocks => (tile, RowChunks::Padded),
            Self::SwizzledBlocks => (tile, RowChunks::Swizzled),
        };
        StageStorage::Tiled { block, chunks }
    }
}

/// Nothing for the strided layout, so the settings that hold it keep their names.
impl core::fmt::Display for StageLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Strided => Ok(()),
            Self::PaddedRows => write!(f, " layout=padded_rows"),
            Self::Blocks => write!(f, " layout=blocks"),
            Self::PaddedBlocks => write!(f, " layout=padded_blocks"),
            Self::SwizzledBlocks => write!(f, " layout=swizzled_blocks"),
        }
    }
}
