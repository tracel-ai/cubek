//! Lowering `c.mma(a, b)`: at a final tile, the leaf instruction; while levels remain,
//! walk this level under its [`Buffering`]. One walk serves every level: what each operand costs
//! is its own [`Residence`], and a level whose operands all stay put buffers a ring of slots that
//! allocate nothing. Register residency is the kernel's explicit bracket
//! ([`promote`](Tile) then [`copy_from`](Tile::copy_from)), not a lowering decision.

use cubecl::prelude::*;

use crate::instruction::mma::mma_leaf;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c.mma(a, b)`: contract at a final tile, else walk this level.
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
