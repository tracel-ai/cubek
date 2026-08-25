//! The output half of [`Residence`], where a residence is read as a *scope* rather than a refill.
//! Both directions share the vocabulary and differ in which way values travel: an input is
//! *filled* from its source into its residence, an output *drains* from its residence into its
//! sink.
//!
//! An output states its residence at the level whose region it lives across, the way an input
//! does. The output's own space has no contracted axis, so its outermost level already holds one
//! region per instance: stating [`Register`](Residence::Register) there opens a register-resident
//! accumulator for that whole region, and every level below runs inside it.
//!
//! ```ignore
//! let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);  // opens the scope
//! acc.mm(&a, &b);                   // `c = a·b`, and the drain that closes the scope
//! ```
//!
//! `mm` states `c = a·b`, so it owns the init. To fold onto what `c` already holds, state that
//! init and use the accumulating verb instead:
//!
//! ```ignore
//! acc.seed();
//! acc.mma(&a, &b);
//! ```
//!
//! An output stating nothing is [`InPlace`](Residence::InPlace) and contracts where it already
//! lies. Both read as those three lines: what differs is the operand's statement, not the kernel.

use cubecl::prelude::*;

use crate::*;

/// An accumulator's scope, opened by [`Tile::accumulate`] and closed by whichever op exhausts it.
///
/// The variants are the output's [`Residence`] at the level the scope opens, not a choice made
/// here. `EA` is the register accumulate element, distinct from the stored `Out` (`f32` accumulate
/// under an `f16` output); an `InPlace` scope has no second element, so it never reads `EA`.
// The register variant carries a second tile; both are expansion-time handles, so the size gap
// costs nothing the kernel ever sees.
#[allow(clippy::large_enum_variant)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum AccumulatorScope<EA: Numeric, Out: Numeric> {
    /// Contracts in a plane-resident partition, draining into the output on the way out.
    Register {
        tile: Tile<EA>,
        sink: Tile<Out>,
        #[cube(comptime)]
        monoid: Monoid,
    },
    /// Contracts in the output itself, so there is nothing to drain.
    InPlace {
        sink: Tile<Out>,
        #[cube(comptime)]
        monoid: Monoid,
    },
}

#[cube]
impl<EA: Numeric, Out: Numeric> AccumulatorScope<EA, Out> {
    /// Seed the accumulator with its monoid's identity: `0` under `Sum`, the lowest value under
    /// `Max`, and so on. The identity is minted in whichever element the accumulator actually
    /// holds, which is the element the caller cannot name here.
    pub fn seed(&mut self) {
        match self {
            AccumulatorScope::Register {
                tile,
                sink: _,
                monoid,
            } => tile.init_identity(comptime!(*monoid)),
            AccumulatorScope::InPlace { sink, monoid } => sink.init_identity(comptime!(*monoid)),
        }
    }

    /// `c = lhs · rhs`: contract into the accumulator and drain, the contraction owning the init.
    /// The scope's whole body at every call site that is not accumulating onto `c`.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        match self {
            AccumulatorScope::Register {
                tile,
                sink,
                monoid: _,
            } => {
                tile.mm(lhs, rhs);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid: _ } => sink.mm(lhs, rhs),
        }
    }

    /// `c += lhs · rhs`: [`mm`](AccumulatorScope::mm) with the accumulate its name carries,
    /// folding onto an accumulator the caller [`seed`](AccumulatorScope::seed)ed. Drains the same
    /// way: the contraction exhausts the scope the accumulator was opened for, so leaving it is
    /// what writes the result back.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        match self {
            AccumulatorScope::Register {
                tile,
                sink,
                monoid: _,
            } => {
                tile.mma(lhs, rhs);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid: _ } => sink.mma(lhs, rhs),
        }
    }

    /// `c = fold(input)`: reduce `input` along the axes the accumulator does not span, then
    /// drain, as [`mm`](AccumulatorScope::mm) does. Owns the init the same way.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                tile.reduce_axis(input, comptime!(*monoid));
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                sink.reduce_axis(input, comptime!(*monoid))
            }
        }
    }

    /// `c = fold(c, input)`: [`reduce_axis`](AccumulatorScope::reduce_axis) folding onto an
    /// accumulator the caller [`seed`](AccumulatorScope::seed)ed.
    pub fn reduce_axis_accumulate<In: Numeric>(&mut self, input: &Tile<In>) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                tile.reduce_axis_accumulate(input, comptime!(*monoid));
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                sink.reduce_axis_accumulate(input, comptime!(*monoid))
            }
        }
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Open this output's accumulator scope, uninitialized, folding under `monoid`: `Sum` for a
    /// matmul, whichever fold a reduce asked for. `monoid` is stated here because it is read on
    /// drain, when the plane's lanes are combined, and comptime state cannot be set after a thing is
    /// built, and because it is the accumulation's one algebra rather than a fact about each
    /// call. The op that closes the scope drains it; where that op is
    /// [`mm`](AccumulatorScope::mm) or [`reduce_axis`](AccumulatorScope::reduce_axis) it owns the
    /// init too, and only the accumulating verbs ask the caller to state one
    /// ([`seed`](AccumulatorScope::seed)).
    ///
    /// Where the accumulator lives is the output operand's own statement, read off the residence
    /// it stated at this level ([`Operand::stage`]). `EA` is the register accumulate type, read
    /// only when that statement is [`Register`](Residence::Register).
    ///
    /// `lhs` is the operand this accumulator will contract against, and it states the
    /// register form the accumulator takes: a staged cmma or manual-mma operand meets a
    /// matching fragment. An operand staging no register form leaves the instruction open (memory
    /// windows serve the software instruction and both hardware mmas alike), so the space's
    /// instruction decides ([`Space::instruction`]). `lhs` also sizes the
    /// fragment: a hardware fragment is the whole `m × n × k` instruction, and an accumulator
    /// spans only `m × n`, so the contraction depth has to come from a side that has it. The
    /// `mma` call is the next line at every call site, so it is already in hand.
    pub fn accumulate<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> AccumulatorScope<EA, Acc> {
        let plan = self.stage_plan();
        match comptime!(plan.head()) {
            Residence::Register => {
                let tile = self.register_partition::<EA, EL>(lhs, monoid);
                AccumulatorScope::<EA, Acc>::new_Register(tile, self.clone(), monoid)
            }
            Residence::InPlace => AccumulatorScope::<EA, Acc>::new_InPlace(self.clone(), monoid),
            Residence::Smem => panic!(
                "Tile::accumulate: an accumulator has no shared-memory form; state \
                 Residence::Register to contract in registers, or nothing to contract in place"
            ),
        }
    }

    /// The plane-resident partition a [`Register`](Residence::Register) scope contracts in,
    /// uninitialized and shaped to meet `lhs` at the instruction.
    fn register_partition<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        // The contracted axes are those `lhs` spans and this accumulator does not, and the
        // fragment is sized by their product. Off the leaf space: a caller holds a level above
        // it, while the fragment's own reader (`Tile::fragment_matrix`) is handed the leaf tile
        // directly, and the two edges have to be the same number.
        let k = comptime!(lhs.space.final_space().contracted_extent(&self.space));
        let form = comptime!(self.space.instruction().expect(
            "Tile::accumulate: the register form is the space's statement; add \
             `.instruction(...)` to its tiling"
        ));
        // The block's lines match the memory it drains back into; the hardware
        // encodings ignore it.
        let vector_size = self.vector_size();
        // Not `self.lane_share()`: that is the stamped value, and stamping happens on the way
        // down, after this runs. The space already knows every level, so ask it.
        let lane_share = comptime!(self.space.leaf_lane_share());
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(form),
            comptime!(k),
            vector_size,
            lane_share,
            monoid,
        )
    }
}
