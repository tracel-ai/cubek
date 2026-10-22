use crate::attention::{B, D, DV, KV, S};
use cubecl::tile::{Axis, Space, StageStorage, stage::RowChunks};

/// How the cube lays each stage of keys and values out in shared memory.
///
/// Where a row of keys or values spans a whole number of the banks' 128 bytes, as a head dim of 64
/// or 128 in `f16` does, the matrix instruction, loading the same columns of several rows at once,
/// reads one bank several times over. Padding each row by a chunk starts the next row on other
/// banks, at a chunk a row; blocking the stage into the instruction's tiles packs one tile's rows
/// into consecutive bytes instead, at none, and swizzling a tile's chunks spreads its rows over the
/// banks in place, also at none. Which reads faster is the device's to say, so each is a setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeAttentionStageLayout {
    /// Row-major over the whole stage, the rows in order.
    Strided,
    /// Row-major over the whole stage, each row one chunk longer.
    PaddedRows,
    /// In tiles of the instruction, one after another, their rows in order.
    Blocks,
    /// In tiles of the instruction, each of their rows one chunk longer.
    PaddedBlocks,
    /// In tiles of the instruction, each row's chunks swizzled: what a fragment reads that hands
    /// every row its own address, as cubek's own transport does; a cmma fragment reads a tile off
    /// a pointer and a stride, and cannot.
    SwizzledBlocks,
}

impl CubeAttentionStageLayout {
    /// The storage a stage of keys and values over `stage` is held in, the instructions' tile
    /// `tile` cells along each of the keys, the head dim and the value dim.
    pub fn storage(self, stage: &Space, tile: [(Axis, usize); 3]) -> StageStorage {
        let whole = [B, KV, S, D, DV]
            .map(|axis| (axis, stage.extent(axis)))
            .to_vec();
        let tiles = [(B, 1), (KV, 1)].into_iter().chain(tile).collect();
        match self {
            Self::Strided => StageStorage::Strided,
            Self::PaddedRows => StageStorage::Tiled {
                block: whole,
                chunks: RowChunks::Padded,
            },
            Self::Blocks => StageStorage::Tiled {
                block: tiles,
                chunks: RowChunks::InOrder,
            },
            Self::PaddedBlocks => StageStorage::Tiled {
                block: tiles,
                chunks: RowChunks::Padded,
            },
            Self::SwizzledBlocks => StageStorage::Tiled {
                block: tiles,
                chunks: RowChunks::Swizzled,
            },
        }
    }
}

/// Nothing for the strided layout, so the settings that hold it keep their names.
impl core::fmt::Display for CubeAttentionStageLayout {
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
