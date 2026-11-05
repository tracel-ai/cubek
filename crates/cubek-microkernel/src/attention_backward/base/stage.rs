use crate::attention::plane::ScoreHolder;
use crate::attention::{B, D, DV, G, KV, Q, S};
use crate::matmul::plane::PlaneFragmentMicrokernel;
use cubecl::tile::StageStorage;
use cubecl::tile::stage::RowChunks;

/// The storage a backward walk holds its stages in, its planes contracting through `score`'s
/// instruction and keeping the score as `holder` says.
///
/// A row of a head of 64 or 128 `f16` spans a whole number of the banks' 128 bytes, so a matrix
/// instruction loading the same columns of several rows of a row-major stage reads one bank
/// several times over. A plane holding its score in manual-mma fragments loads every stage by
/// `ldmatrix`, each row at its own address, so its stages lie in tiles of the instruction's depth
/// along every axis, each row's chunks swizzled over the banks. A plane reading through windows
/// loads cmma fragments, which read a tile off a pointer and a stride, so row-major.
pub fn stage_storage(holder: ScoreHolder, score: PlaneFragmentMicrokernel) -> StageStorage {
    match holder {
        ScoreHolder::Fragments => {
            let [_, _, (_, depth)] = score.instruction.shape();
            let block = [B, KV, G]
                .map(|axis| (axis, 1))
                .into_iter()
                .chain([Q, S, D, DV].map(|axis| (axis, depth)))
                .collect();
            StageStorage::Tiled {
                block,
                chunks: RowChunks::Swizzled,
            }
        }
        ScoreHolder::Window => StageStorage::Strided,
    }
}
