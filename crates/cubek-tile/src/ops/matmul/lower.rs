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

/// A level that distributes work as one, walked as if every axis were dealt on its own, would
/// give each instance the whole grid: the wrong answer computed once per instance. The share is
/// the accumulator scope's ([`Tile::accumulate`]), because the scope is what changes.
fn refuse_distributed_work(space: &Space) {
    assert!(
        space.partitioner().work().is_none(),
        "Tile::mma: this level distributes its axes' work as one, so a contraction here is an \
         instance's share of it rather than the whole grid's. Open the output's accumulator \
         scope (`c.accumulate(..)`) and contract through it."
    );
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b`: contract at a final tile, else walk this level. `c` is a result, so nothing it
    /// held before takes part. Where the leaf owns each output cell outright the register block
    /// starts from the identity instead of reading `c` back, which for a short `kc` is a
    /// measurable share of the leaf; where it does not, the seed happens here anyway.
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
    /// `c` holds; nothing here initializes it. Also the recursion the walk re-enters per region,
    /// deliberately: what the accumulation starts from is decided once at the top, since a
    /// region's operands always span their own leaf and re-deciding would overwrite every step.
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
                comptime!(refuse_distributed_work(&self.space));
                let op_space = self.op_space(lhs, rhs);
                self.mma_buffered(
                    lhs,
                    rhs,
                    Walk::over(op_space),
                    comptime!(level.buffering().depth()),
                    semiring,
                );
            }
        }
    }

    /// [`mma`](Tile::mma) over `steps` of this level's regions starting at `base`, not all of
    /// them. What a streamed instance runs on one output tile: its share of that tile's
    /// contraction is a range of the line, and the walk is where a range is said
    /// ([`Walk::window`]). Everything under it is the contraction a whole region gets.
    pub(crate) fn mma_window<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        base: usize,
        steps: usize,
        #[comptime] semiring: Semiring,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => {
                panic!("Tile::mma_window: a final tile has no walk to take a range of")
            }
            Partitioner::Level(level) => {
                let op_space = self.op_space(lhs, rhs);
                self.mma_buffered(
                    lhs,
                    rhs,
                    Walk::over(op_space).window(base, steps),
                    comptime!(level.buffering().depth()),
                    semiring,
                );
            }
        }
    }

    /// `c = (a ⊗ s) · b`, or `c = a · (b ⊗ s)`: [`mm`](Tile::mm) with one operand scaled by a
    /// real operand.
    ///
    /// The scales are an operand like any other, and the arithmetic that folds them in is this
    /// verb: nothing decodes behind a read.
    ///
    /// **Which operand it scales is not stated**: the scales' own axes say it
    /// ([`ScaleSide`](crate::ScaleSide)). A scale spanning the output's columns is a fact about
    /// the rhs's columns; anything else scales the lhs. Both are the same sum of terms, so one
    /// verb serves both, folding once per `(row, k)` or once per `(col, k)`.
    ///
    /// `s` resolves at whatever granularity its axes give it, and cannot vary over an axis it does
    /// not address. The block is an axis of the problem, `(KB, KI)` or `(NB, NI)`, spelled with
    /// [`PhysicalAxisMap::disjoint`](crate::PhysicalAxisMap::disjoint) on the values while the
    /// scales leave the position inside it unmapped, so no line can straddle a block whatever
    /// width it is served at. A scales operand that divides instead is refused.
    pub fn mm_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
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
        scales: &Sequence<Tile<S>>,
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
                    Walk::over(op_space),
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
    /// no axis beyond `lhs ∪ rhs`, which is why the schedules merge the same two.
    ///
    /// The accumulator is asked for sizes all the same, and first: spanning an axis and being able
    /// to state its size are different things (a gathered operand's bound is the receptive field
    /// its axes reach over, so it answers for neither), and an axis the output spans is one it
    /// writes, so its bound is the extent the walk must cover.
    pub(crate) fn op_space<Lhs: Numeric, Rhs: Numeric>(
        &self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
    ) -> Space {
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
pub(crate) fn mma_leaf_scaled<E: Numeric, EL: Numeric, ER: Numeric, S: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Sequence<Tile<S>>,
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
    /// [`mma`](PlaneTile::mma) with one operand scaled by a real operand. Only the register form:
    /// a hardware instruction eats its operands' format whole, so a scale there routes to the
    /// *fragment* rather than to a view, which is a different instruction.
    pub fn mma_scaled<EL: Numeric, ER: Numeric, ES: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        scales: &Sequence<Tile<ES>>,
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

/// Asserts that operands are not gathered and read as one matrix each. A fragment contracts over
/// one `k` edge, which is not one contracted *axis*: axes carried as one run flatten into an edge,
/// and a partitioned contraction is exactly that. What it cannot read is a contraction its axes
/// give no edge for.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let kc = Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.space);
        MatrixAxes::find(&lhs.space, axes.rows(&out), kc).is_some()
            && MatrixAxes::find(&rhs.space, kc, axes.cols(&out)).is_some()
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

#[cfg(test)]
mod tests {
    use super::refuse_distributed_work;
    use crate::{Axis, Buffering, CubeAxis, Space, Tiling, WalkOrder, cubes};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    // Host-side, because a comptime panic raised in a kernel lands on a worker thread where
    // `#[should_panic]` never sees it and the launch returns zeros.

    fn space(distributed: bool) -> Space {
        Tiling::over(&mut (), &[(M, 8), (N, 8), (K, 8)])
            .level(
                WalkOrder::RowMajor,
                Buffering::SINGLE,
                |l, _| match distributed {
                    true => {
                        l.distribute(cubes(CubeAxis::X).instances(3), &[(M, 4), (N, 4), (K, 8)]);
                    }
                    false => {
                        l.walk(&[(M, 4), (N, 4), (K, 8)]);
                    }
                },
            )
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.walk(&[(M, 4), (N, 4), (K, 4)]);
            })
            .build()
    }

    #[test]
    #[should_panic = "distributes its axes' work as one"]
    fn contracting_distributed_work_whole_is_refused() {
        refuse_distributed_work(&space(true));
    }

    #[test]
    fn contracting_a_level_dealt_per_axis_is_the_walk_it_always_was() {
        refuse_distributed_work(&space(false));
    }
}
