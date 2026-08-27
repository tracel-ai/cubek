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
//! let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);  // opens the scope under its fold
//! acc.mm(&a, &b, Semiring::SUM_PROD);                    // `c = a·b`, and the drain that closes it
//! ```
//!
//! `mm` states `c = a·b`, so it owns the init. To fold onto what `c` already holds, state that
//! init and use the accumulating verb instead:
//!
//! ```ignore
//! acc.seed();
//! acc.mma(&a, &b, Semiring::SUM_PROD);
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

/// A contraction into this accumulation has to add the way the accumulation folds: the scope
/// seeds and drains under its monoid, so a semiring adding under another would merge the plane's
/// partials one way and accumulate the steps another, and the result is neither.
fn adds_the_way_it_folds(monoid: Monoid, semiring: Semiring) {
    assert!(
        semiring.add() == monoid,
        "AccumulatorScope: this accumulation folds under {monoid:?}, so it cannot contract under \
         {semiring:?}"
    );
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
    ///
    /// `semiring` is the algebra of the contraction itself: the product it forms from a pair of
    /// operands and the monoid those products accumulate into. The scope holds a fold and nothing
    /// more, so a contraction, which needs more than a fold, is handed one.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mm(lhs, rhs, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mm(lhs, rhs, semiring)
            }
        }
    }

    /// `c += lhs · rhs`: [`mm`](AccumulatorScope::mm) with the accumulate its name carries,
    /// folding onto an accumulator the caller [`seed`](AccumulatorScope::seed)ed. Drains the same
    /// way: the contraction exhausts the scope the accumulator was opened for, so leaving it is
    /// what writes the result back.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mma(lhs, rhs, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mma(lhs, rhs, semiring)
            }
        }
    }

    /// `c = (lhs ⊗ s) · rhs`, or its rhs twin: [`mm`](AccumulatorScope::mm) with one operand
    /// scaled by a real operand, the side read off the scales' axes
    /// ([`ScaleSide`](crate::ScaleSide)).
    ///
    /// A register accumulator here is what a decode gemv wants: the scaled partials never
    /// round-trip through the sink between `K` steps.
    pub fn mm_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Tile<S>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mm_scaled(lhs, rhs, scales, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mm_scaled(lhs, rhs, scales, semiring)
            }
        }
    }

    /// `c += (lhs ⊗ s) · rhs`: [`mm_scaled`](AccumulatorScope::mm_scaled) with the accumulate its
    /// name carries, folding onto an accumulator the caller [`seed`](AccumulatorScope::seed)ed.
    pub fn mma_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Tile<S>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mma_scaled(lhs, rhs, scales, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mma_scaled(lhs, rhs, scales, semiring)
            }
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
    /// matmul, `Min` for a min-plus one, whichever fold a reduce asked for. `monoid` is stated
    /// here because it is read on drain, when the plane's lanes are combined, and comptime state
    /// cannot be set after a thing is built, and because it is the accumulation's one algebra
    /// rather than a fact about each call. It is a fold and nothing more: an op needing more,
    /// [`mm`](AccumulatorScope::mm) and [`mma`](AccumulatorScope::mma), is handed the
    /// [`Semiring`] it runs, whose add this must be.
    ///
    /// The op that closes the scope drains it; where that op is [`mm`](AccumulatorScope::mm) or
    /// [`reduce_axis`](AccumulatorScope::reduce_axis) it owns the init too, and only the
    /// accumulating verbs ask the caller to state one ([`seed`](AccumulatorScope::seed)).
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
