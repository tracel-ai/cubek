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
    /// [`copy_from`](Tile::copy_from) after. What the register form *is* belongs to the
    /// declared [`Leaf`](crate::Leaf) — each instruction tells its own backing store.
    pub fn promote(&self) -> Tile<Acc> {
        let leaf = comptime!(self.space.partitioner().leaf());
        let mut acc = match comptime!(leaf) {
            Leaf::Cmma { k } => CmmaPartition::mirror(comptime!(self.space.clone()), k),
            Leaf::Register => {
                panic!("Tile::promote: the register leaf runs in place — nothing to promote")
            }
        };
        acc.copy_from(self);
        acc
    }
}
