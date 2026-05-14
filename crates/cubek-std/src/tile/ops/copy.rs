use cubecl::prelude::*;

use crate::{
    StageIdent,
    tile::{Tile, TileExpand, TileKind, TileKindExpand, TileScope},
};

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc, ReadWrite> {
    /// Zero-initializes the tile in place using the per-variant init that
    /// matches the storage's expected layout. Each arm delegates to the
    /// variant's `init_zero` method.
    ///
    /// `L` / `R` are only consulted on the MMA path (which needs the matmul
    /// type triple at the layout-aware load); other variants ignore them.
    pub fn init_zero<L: Numeric, R: Numeric>(&mut self, #[comptime] ident: StageIdent) {
        match &mut self.kind {
            TileKind::Cmma(t) => t.init_zero(),
            TileKind::Bounce(b) => b.init_zero(),
            TileKind::Mma(t) => t.init_zero::<L, R>(),
            TileKind::Register(t) => t.init_zero(ident),
            TileKind::PlaneVec(t) => t.init_zero(),
            TileKind::Interleaved(t) => t.init_zero(),
            _ => panic!("init_zero: unsupported tile variant"),
        }
    }

    /// Copies data from `source` into `self`. Each arm delegates to the
    /// destination variant's `copy_from` method (where the per-variant
    /// load/zero-init/write helpers live).
    ///
    /// `SS` is the vector size of the shared memory tile involved in the
    /// copy (whether that's the source on a load, or the destination on a
    /// write). `L`/`R`/`A` are the matmul-level numeric types needed by the
    /// MMA readers/writers — they are unused on non-MMA paths.
    pub fn copy_from<
        SE: Numeric,
        SS: Size,
        L: Numeric,
        R: Numeric,
        A: Numeric,
        SIO: SliceVisibility,
    >(
        &mut self,
        source: &Tile<SE, Sc, SIO>,
        #[comptime] ident: StageIdent,
    ) {
        match &mut self.kind {
            TileKind::Cmma(t) => t.copy_from::<SE, SS, Sc, SIO>(source, ident),
            TileKind::Bounce(b) => b.copy_from::<SE, SS, Sc, SIO>(source, ident),
            TileKind::Mma(t) => t.copy_from::<SE, SS, L, R, A, Sc, SIO>(source, ident),
            TileKind::Register(t) => t.copy_from::<SE, SS, Sc, SIO>(source, ident),
            TileKind::PlaneVec(t) => t.copy_from::<SE, SS, Sc, SIO>(source, ident),
            TileKind::Interleaved(t) => t.copy_from::<SE, SS, Sc, SIO>(source, ident),
            TileKind::SharedMemory(shared) => {
                shared.copy_from::<SE, SS, L, R, Sc, SIO>(source);
            }
            _ => panic!("copy_from: unsupported destination variant"),
        }
    }
}
