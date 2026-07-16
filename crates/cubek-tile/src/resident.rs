//! Register residency: the kernel brackets the whole operation, where [`Staging`]
//! refills per region.
//!
//! ```ignore
//! let mut acc = c.promote();  // register form, uninitialized
//! acc.zero();                 // init_accumulator (or acc.copy_from(&c) to accumulate)
//! acc.mma(&a, &b);            // the whole contraction, register-resident
//! c.copy_from(&acc);          // epilogue
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form: pure change of residence, the
    /// fragments uninitialized. The caller states the init ([`zero`](Tile::zero) or
    /// [`copy_from`](Tile::copy_from)) and writes back with `copy_from` after. The
    /// register form itself belongs to the declared [`Leaf`](crate::Leaf).
    pub fn promote(&self) -> Tile<Acc> {
        let leaf = comptime!(self.space.partitioner().leaf());
        match comptime!(leaf) {
            Leaf::Cmma { k } => CmmaPartition::mirror(comptime!(self.space.clone()), k),
            Leaf::Register => {
                panic!("Tile::promote: the register leaf runs in place — nothing to promote")
            }
        }
    }
}
