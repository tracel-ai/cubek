use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{Tile, TileExpand, TileKind, TileKindExpand, TileScope},
};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    /// Zero-initialize the tile in place. `L`/`R` are only consulted on MMA.
    pub fn init_zero<L: Numeric, R: Numeric>(&mut self, #[comptime] ident: StageIdent) {
        match &mut self.kind {
            TileKind::Cmma(t) => t.init_zero(),
            TileKind::Bounce(b) => b.init_zero(),
            TileKind::Mma(t) => t.init_zero::<L, R>(),
            TileKind::Register(t) => t.init_zero(ident),
            TileKind::PlaneVec(t) => t.init_zero(),
            TileKind::Interleaved(t) => t.init_zero(),
            TileKind::RowWise(t) => t.init_zero(),
            _ => panic!("init_zero: unsupported tile variant"),
        }
    }

    /// Copy `source` into `self`. `SS` is the smem vector size involved in
    /// the copy; `L`/`R`/`A` are only consulted on MMA paths.
    pub fn copy_from<SE: Numeric, SS: Size, L: Numeric, R: Numeric, A: Numeric>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] ident: StageIdent,
    ) {
        match &mut self.kind {
            TileKind::Cmma(t) => t.copy_from::<SE, SS, Sc>(source, ident),
            TileKind::Bounce(b) => b.copy_from::<SE, SS, Sc>(source, ident),
            TileKind::Mma(t) => t.copy_from::<SE, SS, L, R, A, Sc>(source, ident),
            TileKind::Register(t) => t.copy_from::<SE, SS, Sc>(source, ident),
            TileKind::PlaneVec(t) => t.copy_from::<SE, SS, Sc>(source, ident),
            TileKind::Interleaved(t) => t.copy_from::<SE, SS, Sc>(source, ident),
            TileKind::SharedTile(shared) => {
                shared.copy_from::<SE, SS, L, R, Sc>(source);
            }
            _ => panic!("copy_from: unsupported destination variant"),
        }
    }
}
