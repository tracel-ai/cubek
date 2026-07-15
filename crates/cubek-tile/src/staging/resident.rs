//! Register residency: the kernel brackets the whole operation, where [`Staging`]
//! refills per region.
//!
//! ```ignore
//! let mut acc = c.promote::<EA>();  // zeroed register accumulator (EA = f32)
//! acc.mma(&a, &b);                  // the whole contraction, register-resident
//! acc.drain_cast_into(&mut c);      // epilogue, casting EA down to c's element type
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form, zeroed (`c = a·b`, so `beta = 0`);
    /// write it back with [`drain_cast_into`](Tile::drain_cast_into) after. `EA` is the
    /// register accumulate type, distinct from the served/stored `Acc` (e.g. `f32`
    /// accumulate under an `f16` output). The register form itself belongs to the declared
    /// [`Leaf`](crate::Leaf).
    pub fn promote<EA: Numeric>(&self) -> Tile<EA> {
        let leaf = comptime!(self.space.partitioner().leaf());
        match comptime!(leaf) {
            Leaf::Cmma { k } => CmmaPartition::<EA>::mirror(comptime!(self.space.clone()), k),
            Leaf::Register => {
                panic!("Tile::promote: the register leaf runs in place — nothing to promote")
            }
        }
    }
}
