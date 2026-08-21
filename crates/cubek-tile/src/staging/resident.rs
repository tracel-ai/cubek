//! The output half of [`Residence`]: an accumulator moved to
//! [`Register`](Residence::Register) for the whole operation, rather than an input refilled per region
//! by [`Staging`]. The two directions share the vocabulary and differ in which way values travel:
//! an input is *filled* from its source into its residence, an output *drains* from its residence
//! into its sink. Where an input states its residences per level and the walk acts on them, an
//! accumulator's is the kernel's own explicit bracket, since only the kernel knows what initializes
//! it and when the result is wanted back.
//!
//! ```ignore
//! let mut acc = c.promote::<EA, _>(&a);  // register form, uninitialized (EA = f32)
//! acc.zero();                       // init (or acc.copy_from(&c) to accumulate)
//! acc.mma(&a, &b);                  // the whole contraction, register-resident
//! acc.drain_cast_into(&mut c);      // epilogue, casting EA down to c's element type
//! ```

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form ([`Residence::Register`]): pure change of
    /// residence, the fragments uninitialized. The caller states the init ([`zero`](Tile::zero) for
    /// `c = a·b`, or [`copy_from`](Tile::copy_from) to accumulate) and writes it back with
    /// [`drain_cast_into`](Tile::drain_cast_into) after. `EA` is the register accumulate
    /// type, distinct from the served/stored `Acc` (e.g. `f32` accumulate under an `f16`
    /// output).
    ///
    /// `lhs` is the operand this accumulator will contract against, and it states the
    /// register form the accumulator takes: a staged cmma or manual-mma operand meets a
    /// matching fragment. An operand staging no register form leaves the instruction open (memory
    /// windows serve the software instruction and both hardware mmas alike), so the space's
    /// floor decides ([`Space::instruction`]). `lhs` also sizes the
    /// fragment: a hardware fragment is the whole `m × n × k` instruction, and an accumulator
    /// spans only `m × n`, so the contraction depth has to come from a side that has it. The
    /// `mma` call is the next line at every call site, so it is already in hand.
    pub fn promote<EA: Numeric, EL: Numeric>(&self, lhs: &Tile<EL>) -> Tile<EA> {
        // The contracted axes are those `lhs` spans and this accumulator does not, and the fragment
        // is sized by their product. Off the leaf space: a caller holds a level above it, while the
        // fragment's own reader (`Tile::fragment_matrix`) is handed the leaf tile directly, and the
        // two edges have to be the same number.
        let k = comptime!(lhs.space.final_space().contracted_extent(&self.space));
        let form = comptime!(self.space.instruction().expect(
            "Tile::promote: the register form is the space's statement; add `.instruction(...)` \
             to its tiling"
        ));
        // The block's lines match the memory it drains back into; the hardware
        // encodings ignore it.
        let vector_size = self.vector_size();
        let lane_share = self.lane_share();
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(form),
            comptime!(k),
            vector_size,
            lane_share,
        )
    }
}
