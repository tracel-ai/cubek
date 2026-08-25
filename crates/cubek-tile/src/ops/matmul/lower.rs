//! Lowering `c.mm(a, b)` and `c.mma(a, b)`: at a final tile, the leaf dispatch ([`mma_leaf`]);
//! while levels remain, walk this level under its [`Buffering`]. One walk serves every level: what each operand costs
//! is its own [`Residence`], and a level whose operands all stay put buffers a ring of slots that
//! allocate nothing. An accumulator's register residency is read the same way, by
//! [`accumulate`](Tile::accumulate) opening the scope it states, not by a lowering decision.

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
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let spans = self.contracts_whole_at_leaf(lhs, rhs);
        let init_from = self.request_init_from(comptime!(spans));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(Semiring::SUM_PROD.add())),
        }
        self.mma(lhs, rhs);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c += a · b`: [`mm`](Tile::mm) with the accumulate its name carries. Folds onto whatever
    /// `c` holds, which the caller owns: nothing here initializes it.
    ///
    /// Also the recursion the walk re-enters per region, and deliberately so: what the
    /// accumulation starts from is decided once, at the top, from the undivided operand spaces. A
    /// region's operands are one contracted step of the whole and always span their own leaf, so
    /// re-deciding down here would overwrite at every step of a walk that must fold them together.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => mma_leaf(self, lhs, rhs),
            Partitioner::Level(level) => {
                let op_space = self.op_space(lhs, rhs);
                self.mma_buffered(lhs, rhs, op_space, comptime!(level.buffering().depth()));
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
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space),
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space)
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
            contract::memory::<E, EL, ER>(g, lhs, rhs, space, config)
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => panic!("mma: a procedural tile is not an accumulator sink"),
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
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                strided_2d(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
            PlaneTile::Mma(d) => {
                flattened_k(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
        }
    }
}

/// Asserts that operands are not gathered and have a single contracted axis.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered
            && !rhs_gathered
            && Space::contracted(&[&lhs.space, &rhs.space], &out).len() == 1,
        "mma: a cmma or plane-register fragment reads one contracted axis off a directly \
         addressed operand; a gather or a wider reduce needs the manual-mma leaf, or an \
         unpromoted Gmem/Smem accumulator, whose software instruction is the \
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
