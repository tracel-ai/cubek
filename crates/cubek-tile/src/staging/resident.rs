//! Register residency: the kernel brackets the whole operation, where [`Staging`]
//! refills per region.
//!
//! ```ignore
//! let mut acc = c.promote::<EA>();  // register form, uninitialized (EA = f32)
//! acc.zero();                       // init (or acc.copy_from(&c) to accumulate)
//! acc.mma(&a, &b);                  // the whole contraction, register-resident
//! acc.drain_cast_into(&mut c);      // epilogue, casting EA down to c's element type
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form: pure change of residence, the
    /// fragments uninitialized. The caller states the init ([`zero`](Tile::zero) for
    /// `c = a·b`, or [`copy_from`](Tile::copy_from) to accumulate) and writes it back with
    /// [`drain_cast_into`](Tile::drain_cast_into) after. `EA` is the register accumulate
    /// type, distinct from the served/stored `Acc` (e.g. `f32` accumulate under an `f16`
    /// output). The register form itself belongs to the declared [`Leaf`](crate::Leaf).
    pub fn promote<EA: Numeric>(&self) -> Tile<EA> {
        PlanePartition::<EA>::mirror(comptime!(self.space.clone()))
    }
}
