//! Lowering `c.mm(a, b)` and `c.mma(a, b)`: at a final tile, the leaf dispatch ([`mma_leaf`]);
//! while levels remain, walk this level under its [`Buffering`]. One walk serves every level: what each operand costs
//! is its own [`Residence`], and a level whose operands all stay put buffers a ring of slots that
//! allocate nothing. An accumulator's register residency is read the same way, by
//! [`accumulate`](Tile::accumulate) opening the scope it states, not by a lowering decision.
//!
//! The [`Semiring`] rides down the same way: the accumulation states its algebra once and every
//! level hands the same one to the leaf that runs the steps.

use cubecl::prelude::*;

use crate::instruction::registers::contract;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b`: contract at a final tile, else walk this level. `c` is a result, so nothing
    /// it held before takes part.
    ///
    /// Where the leaf owns each output cell outright, that lets the register block start from the
    /// identity instead of reading `c` back, which is one load per cell and, for a `kc` too short
    /// to amortize it, a measurable share of the leaf. Where it does not, the seed the caller
    /// would have written happens here instead, so the verb costs nothing to reach for.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        let spans = self.contracts_whole_at_leaf(lhs, rhs);
        let init_from = self.request_init_from(comptime!(spans));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
        }
        self.mma(lhs, rhs, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c += a · b`: [`mm`](Tile::mm) with the accumulate its name carries. Folds onto whatever
    /// `c` holds, which the caller owns: nothing here initializes it.
    ///
    /// Also the recursion the walk re-enters per region, and deliberately so: what the
    /// accumulation starts from is decided once, at the top, from the undivided operand spaces. A
    /// region's operands are one contracted step of the whole and always span their own leaf, so
    /// re-deciding down here would overwrite at every step of a walk that must fold them together.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => mma_leaf(self, lhs, rhs, semiring),
            Partitioner::Level(level) => {
                let op_space = self.op_space(lhs, rhs);
                self.mma_buffered(
                    lhs,
                    rhs,
                    op_space,
                    comptime!(level.buffering().depth()),
                    semiring,
                );
            }
        }
    }

    /// `c = (a ⊗ s) · b`, or `c = a · (b ⊗ s)`: [`mm`](Tile::mm) with one operand scaled by a
    /// real operand.
    ///
    /// The scales are an operand like any other — their own tensor, their own axes, named at the
    /// call — and the arithmetic that folds them in is this verb. Nothing decodes behind a read:
    /// an operand that arrives quantized arrives as what it is, and what it takes to serve it is
    /// written here.
    ///
    /// **Which operand it scales is not stated**: the scales' own axes say it
    /// ([`ScaleSide`](crate::ScaleSide)). A scale spanning the output's columns is a fact about
    /// the rhs's columns and nothing else could fold it in; anything else scales the lhs. The two
    /// are the same sum of terms — the scale is one more factor of each — so one verb serves both,
    /// folding once per `(row, k)` or once per `(col, k)`, whichever the operand asks for.
    ///
    /// `s` resolves at whatever granularity its own axes give it, and an axis it does not address
    /// it cannot vary over. Where the block is an axis of the problem — `(KB, KI)`, spelled with
    /// [`PhysicalAxisMap::disjoint`](crate::PhysicalAxisMap::disjoint) on the values — the scales
    /// leave `KI` unmapped and one scale per block follows, with nothing dividing anything. Where
    /// it is not, a [rational](crate::Projection) axis (`PhysicalAxisMap::of(K).over(block)`)
    /// spells the same granularity arithmetically, and then a served line may not straddle a
    /// block: state the cut that holds it.
    pub fn mm_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Tile<S>,
        #[comptime] semiring: Semiring,
    ) {
        let spans = self.contracts_whole_at_leaf(lhs, rhs);
        let init_from = self.request_init_from(comptime!(spans));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
        }
        self.mma_scaled(lhs, rhs, scales, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c += (a ⊗ s) · b` (or its rhs twin): [`mma`](Tile::mma)'s scaled form, and the recursion
    /// the walk re-enters per region.
    pub fn mma_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Tile<S>,
        #[comptime] semiring: Semiring,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => mma_leaf_scaled(self, lhs, rhs, scales, semiring),
            Partitioner::Level(level) => {
                let op_space = self.op_space(lhs, rhs);
                self.mma_scaled_buffered(
                    lhs,
                    rhs,
                    scales,
                    op_space,
                    comptime!(level.buffering().depth()),
                    semiring,
                );
            }
        }
    }

    /// Whether the final tile spans every contracted axis whole, so the walk above the leaf never
    /// returns to a cell it has already written and the one visit that owns a cell may write it
    /// outright.
    fn contracts_whole_at_leaf<Lhs: Numeric, Rhs: Numeric>(
        &self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
    ) -> comptime_type!(InitFrom) {
        comptime!(match Space::merge(&[&lhs.space, &rhs.space])
            .spans_contracted_at_leaf(&self.space)
        {
            true => InitFrom::Identity,
            false => InitFrom::Cell,
        })
    }

    /// The level's operation space: the merge of the operands' spaces, sized by whichever operand
    /// [`witnesses`](Tile::witnesses) each [`Dynamic`](crate::Extent) axis. The output contributes
    /// no axis beyond `lhs ∪ rhs`, which is why the schedules can merge the same two for their own
    /// comptime decisions.
    ///
    /// The accumulator is asked for sizes all the same, because spanning an axis and being able to
    /// state its size are different things. A gathered operand spans the axes its affine map reads
    /// (a convolution's `OH` and `RH` both address one input dim), but its bound is the receptive
    /// field they reach over, so it can answer for neither: the output positions come off the
    /// accumulator and the window off the weights.
    ///
    /// It is asked first for the same reason: an axis it spans is one it writes, so its bound is
    /// the extent the walk must cover, whatever an input's buffer reaches over.
    fn op_space<Lhs: Numeric, Rhs: Numeric>(&self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) -> Space {
        let merged = comptime!({
            let merged = Space::merge(&[&lhs.space, &rhs.space]);
            assert!(
                self.space.axes().all(|axis| merged.contains(axis)),
                "Tile::mma: the output spans an axis neither operand does, so the walk would never \
                 step it and every region would write the same slice"
            );
            merged
        });
        witnessed_space(merged, self, lhs, rhs)
    }
}

/// The leaf contraction `acc += lhs · rhs`. Dispatch is dynamic on the accumulator's comptime
/// storage config
#[cube]
pub fn mma_leaf<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space, semiring),
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space, semiring)
        }
        // A memory accumulator runs the software instruction, under the config the kernel bound
        // on it. A plane-form accumulator that was never promoted lands in the arms above and
        // meets their kind-pairing panics; there is no second declaration left to check this one
        // against.
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            let config = comptime!(match space.instruction() {
                Some(Instruction::Registers { config }) => config,
                Some(other) => panic!(
                    "mma_leaf: a Gmem/Smem accumulator contracts in place through the software \
                     instruction, but the space's instruction is {other:?}; state \
                     Residence::Register on the output so its accumulator is register-resident"
                ),
                None => panic!(
                    "mma_leaf: a Gmem/Smem accumulator contracts through the software \
                     instruction; state it on the space with \
                     `.instruction(Instruction::Registers {{ config }})`"
                ),
            });
            contract::memory::<E, EL, ER>(g, lhs, rhs, space, config, semiring)
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => panic!("mma: a procedural tile is not an accumulator sink"),
    }
}

/// [`mma_leaf`] with one operand scaled: the two accumulators that run the software instruction,
/// in memory or promoted to registers, which are the forms whose step has a scale to apply. A
/// fragment accumulator contracts through a hardware instruction that takes two operands and no
/// scales, so a scaled contraction there is a different instruction, not this one under a flag.
#[cube]
pub fn mma_leaf_scaled<E: Numeric, EL: Numeric, ER: Numeric, S: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Tile<S>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        // A promoted register accumulator: the partials stay in `E` across the whole walk, which
        // is the form a decode gemv wants.
        TileKind::PlaneTile(t) => t.mma_scaled(lhs, rhs, scales, space, semiring),
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf_scaled: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma_scaled(lhs, rhs, scales, space, semiring)
        }
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            let config = comptime!(match space.instruction() {
                Some(Instruction::Registers { config }) => config,
                other => panic!(
                    "mma_leaf_scaled: a scaled contraction runs the software instruction; the \
                     space's instruction is {other:?}. State \
                     `.instruction(Instruction::Registers {{ config }})`"
                ),
            });
            contract::memory_scaled::<E, EL, ER, S>(g, lhs, rhs, scales, space, config, semiring)
        }
        TileKind::TmaGmem(_) => panic!("mma_scaled: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => {
            panic!("mma_scaled: a procedural tile is not an accumulator sink")
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile.
    pub fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                strided_2d(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Mma(d) => {
                flattened_k(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, comptime!(out.clone()));
                d.mma(lhs, rhs, out, semiring)
            }
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// [`mma`](PlaneTile::mma) with one operand scaled by a real operand.
    ///
    /// Only the register form. A hardware instruction eats its operands' format whole, so a scale
    /// there routes to the *fragment*, not to a view: a different instruction, not this one with
    /// a multiply added.
    pub fn mma_scaled<EL: Numeric, ER: Numeric, ES: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        scales: &Tile<ES>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, comptime!(out.clone()));
                d.mma_scaled(lhs, rhs, scales, out, semiring)
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                "mma_scaled: a hardware instruction eats its operands' format, so a scaled                  contraction on a fragment accumulator needs a scaled hardware instruction"
            ),
        }
    }
}

/// Asserts that the algebra is the one a hardware instruction implements: it multiplies and adds.
#[cube]
fn hardware_semiring(#[comptime] semiring: Semiring) {
    comptime!(assert!(
        semiring == Semiring::SUM_PROD,
        "mma: a hardware instruction contracts under the sum-product semiring alone, not \
         {semiring:?}; contract in memory or in a register block to fold under another"
    ));
}

/// Asserts that operands are not gathered and read as one matrix each.
///
/// A fragment contracts over one `k` edge, which is not the same as one contracted *axis*: axes
/// the operand carries as one run flatten into an edge, and a partitioned contraction is exactly
/// that. What it cannot read is a contraction its axes give no edge for.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let rank = out.rank();
        let kc = Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out);
        MatrixAxes::find(&lhs.space, out.extent_at(rank - 2), kc).is_some()
            && MatrixAxes::find(&rhs.space, kc, out.extent_at(rank - 1)).is_some()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `contract::memory` arm of `mma_leaf`"
    ));
}

/// Asserts that operands contract their shared axes in the same order.
#[cube]
fn flattened_k<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.space, &rhs.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.space.contracting(&out),
        rhs.space.contracting(&out)
    ));
}
