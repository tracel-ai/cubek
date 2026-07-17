//! The manual-mma leaf: `acc += lhs · rhs` via [`MmaDefinition::execute`](cubecl::cmma::MmaDefinition),
//! the raw-mma twin of [`cmma`](super::cmma). The accumulator is a register fragment; the operands
//! arrive as fragments or as row-major smem windows, the latter loaded into transient `A`/`B`
//! fragments here (a gmem window's layout is unchecked, so it must be staged first, exactly as in
//! the cmma path).

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<A: Numeric> MmaData<A> {
    /// Manual contraction `self += lhs · rhs`. Fragment operands execute directly; smem operands
    /// are loaded into transient `A`/`B` fragments each call (the engine lays stages out row-major,
    /// which the load assumes).
    pub(crate) fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &Tile<L>, rhs: &Tile<R>) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let io = comptime!(self.io);

        match &mut self.fragment {
            MmaFragment::Acc(acc) => match (&lhs.tile_kind, &rhs.tile_kind) {
                (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                    (PlaneTile::Mma(a), PlaneTile::Mma(b)) => match (&a.fragment, &b.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => {
                            mma_execute::<L, R, A>(af, bf, acc, m, n, k)
                        }
                        _ => panic!("MmaData::mma: operands must carry the Lhs/Rhs roles"),
                    },
                    _ => panic!("MmaData::mma: operands must be mma fragments"),
                },
                (TileKind::Smem(a), TileKind::Smem(b)) => {
                    let mut la = MmaData::<L>::lhs(m, n, k, layout, io);
                    la.load_window(a);
                    let mut rb = MmaData::<R>::rhs(m, n, k, layout, io);
                    rb.load_window(b);
                    match (&la.fragment, &rb.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => {
                            mma_execute::<L, R, A>(af, bf, acc, m, n, k)
                        }
                        _ => panic!("MmaData::mma: transient operand fragments mis-roled"),
                    }
                }
                _ => panic!("MmaData::mma: operands must be fragments or staged smem windows"),
            },
            MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                panic!("MmaData::mma: the accumulator must carry the Acc role")
            }
        }
    }
}
