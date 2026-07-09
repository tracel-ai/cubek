//! Register residency: the kernel brackets the whole operation, where [`Staging`]
//! refills per region.
//!
//! ```ignore
//! let mut acc = c.promote();  // init_accumulator
//! acc.mma(&a, &b);            // the whole contraction, register-resident
//! c.copy_from(&acc);          // epilogue
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form, initialized from the delivered
    /// values; write it back with [`copy_from`](Tile::copy_from) after. The register
    /// form itself belongs to the declared [`Leaf`](crate::Leaf).
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
