//! Register residency: [`promote`](Tile) is [`Staging`](crate::Staging)'s write-side
//! dual. A memory accumulator whose operation runs register-resident is initialized from
//! the accumulator once, accumulated across the whole walk, and written back once — the
//! classic global matmul's `init_accumulator` / epilogue, spelled plainly by the kernel:
//!
//! ```ignore
//! let mut acc = c.promote();  // init_accumulator
//! acc.mma(&a, &b);            // the whole contraction, register-resident
//! c.copy_from(&acc);          // epilogue
//! ```
//!
//! Where `Staging` refills per region, residency brackets the whole operation — the
//! *kernel* enters it, at the outermost scope; the lowering never decides it. What
//! "register form" means (the fragment grid, its windows) is the backing store's business
//! ([`CmmaPartition::mirror`], [`copy_from`](Tile::copy_from)), not ours.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form, initialized from the delivered
    /// values so the operation accumulates onto them; write it back with
    /// [`copy_from`](Tile::copy_from) after. The instruction shape comes from this tile's
    /// own space: `m`/`n` from the final tile, `k` from the declared [`Leaf`](crate::Leaf).
    pub fn promote(&self) -> Tile<Acc> {
        let k = comptime!(self.space.partitioner().leaf().k());
        let mut acc = CmmaPartition::mirror(comptime!(self.space.clone()), k);
        acc.copy_from(self);
        acc
    }
}
