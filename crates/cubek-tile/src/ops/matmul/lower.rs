//! Lowering `c.mma(a, b)`: at a final tile, the leaf dispatch ([`mma_leaf`]); while levels remain,
//! walk this level under its [`Buffering`]. One walk serves every level: what each operand costs
//! is its own [`Residence`], and a level whose operands all stay put buffers a ring of slots that
//! allocate nothing. Register residency is the kernel's explicit bracket
//! ([`promote`](Tile) then [`copy_from`](Tile::copy_from)), not a lowering decision.

use cubecl::prelude::*;

use crate::microkernel::contract;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.mma(a, b)`: contract at a final tile, else walk this level.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        self.mma_with(lhs, rhs, comptime!(false));
    }

    /// `c.mma_replace(a, b)`: [`mma`](Tile::mma) seeding each accumulator cell from the
    /// contraction's identity instead of reading the sink back, so the caller does not zero it
    /// first.
    ///
    /// A memory accumulator round-trips every cell through its store: the leaf seeds a register
    /// block from it ([`AccumulateView::seed`](crate::AccumulateView)) and commits back, and the
    /// zero that made the seed meaningful is a third touch. Replacing drops one load and one store
    /// per cell, which is the whole difference on a contraction with too few taps to amortize
    /// them. A promoted accumulator already states its own init and has nothing to replace.
    ///
    /// Sound only where each cell is contracted once, so the final tile has to span every
    /// contracted axis whole: an axis a level above splits sends the walk back to a cell it has
    /// already written, and the second visit would discard the first. Checked, not trusted.
    pub fn mma_replace<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        comptime!(assert_contracted_at_leaf(
            &self.space,
            &lhs.space,
            &rhs.space
        ));
        self.mma_with(lhs, rhs, comptime!(true));
    }

    pub(crate) fn mma_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] replace: bool,
    ) {
        let partitioner = comptime!(self.space.partitioner().clone());
        match comptime!(partitioner) {
            Partitioner::Final => mma_leaf(self, lhs, rhs, replace),
            Partitioner::Level(level) => {
                let op_space = self.op_space(lhs, rhs);
                self.mma_buffered(
                    lhs,
                    rhs,
                    op_space,
                    comptime!(level.buffering().depth()),
                    replace,
                );
            }
        }
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
    #[comptime] replace: bool,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        TileKind::PlaneTile(t) => {
            comptime!(assert!(
                !replace,
                "mma_leaf: a promoted accumulator states its own init (`zero` for `c = a·b`, \
                 `copy_from` to accumulate), so it has no sink read to replace"
            ));
            t.mma(lhs, rhs, space)
        }
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                !replace,
                "mma_leaf: a promoted accumulator states its own init (`zero` for `c = a·b`, \
                 `copy_from` to accumulate), so it has no sink read to replace"
            ));
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space)
        }
        // A memory accumulator runs the software microkernel. A plane-form accumulator that was
        // never promoted lands in the arms above and meets their kind-pairing panics; there is no
        // second declaration left to check this one against.
        TileKind::Gmem(g) | TileKind::Smem(g) => {
            let config = comptime!(match acc.leaf {
                Leaf::Memory { config } => config,
                _ => panic!("mma_leaf: unpromoted Gmem/Smem accumulator must carry Leaf::Memory"),
            });
            contract::memory::<E, EL, ER>(g, lhs, rhs, space, config, replace)
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

/// Asserts that the final tile spans every contracted axis whole, which is what makes a replacing
/// seed sound: the walk above the leaf then steps only axes the accumulator spans, so it never
/// returns to a cell it has already written.
fn assert_contracted_at_leaf(out: &Space, lhs: &Space, rhs: &Space) {
    let merged = Space::merge(&[lhs, rhs]);
    let leaf = merged.final_space();
    for axis in Space::contracted(&[lhs, rhs], out).iter() {
        let (whole, at_leaf) = (merged.extent(*axis), leaf.extent(*axis));
        assert!(
            whole == at_leaf,
            "Tile::mma_replace: {axis:?} is contracted over {whole} but the final tile spans \
             {at_leaf} of it, so the walk revisits every accumulator cell and a replacing seed \
             would discard what the previous visit left; use `mma` over a zeroed sink"
        );
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
         unpromoted Gmem/Smem accumulator, whose software microkernel is the \
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
