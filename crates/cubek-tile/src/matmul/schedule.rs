//! The walks behind [`Tile::mma`](super::Tile), one per [`Schedule`]. Each schedule walks
//! at the tier its accumulator lives in: memory accumulators walk regions at runtime and
//! stage into shared memory; a fragment-partition accumulator at its partition level walks
//! the register tier — same schedule, but fragments are comptime-indexed, so the walk is
//! static ([`StaticWalk`]) and needs no rendezvous (registers are private).

use cubecl::{cmma::MatrixIdent, prelude::*};

use crate::*;

/// The tier a level's walk runs at. An accumulator knows its own: `Register` when it is
/// a fragment partition at its partition level (fragments are comptime-indexed, so that
/// walk is static), `Memory` otherwise.
enum Tier {
    Memory,
    Register,
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// The [`Tier`] this accumulator's level walks. Comptime self-knowledge — callers
    /// tell the accumulator to walk; they never pick a tier for it.
    fn tier(&self) -> comptime_type!(Tier) {
        match &self.tile_kind {
            TileKind::CmmaPartition(_) => {
                comptime!(if partition_level(&self.space).is_some() {
                    Tier::Register
                } else {
                    Tier::Memory
                })
            }
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                comptime!(Tier::Memory)
            }
        }
    }

    /// `Direct` on this accumulator: no staging — every read goes to where the operand lives.
    pub(crate) fn mma_direct<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        let tier = self.tier();
        match comptime!(tier) {
            Tier::Register => direct_register(lhs, rhs, self),
            Tier::Memory => {
                for region in Walk::over(space) {
                    self.at(&region).mma(&lhs.at(&region), &rhs.at(&region));
                }
            }
        }
    }

    /// `Staged` on this accumulator: per region, stage each operand into the faster tier,
    /// then recurse.
    pub(crate) fn mma_staged<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        let tier = self.tier();
        match comptime!(tier) {
            Tier::Register => staged_register(lhs, rhs, self),
            Tier::Memory => staged_smem(lhs, rhs, self, space),
        }
    }

    /// `DoubleBuffered` on this accumulator. Register double buffering is software
    /// pipelining, which the Metal compiler already does across fragment loads and
    /// executes, so it is not wired.
    pub(crate) fn mma_double<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        space: Space,
    ) {
        let tier = self.tier();
        match comptime!(tier) {
            Tier::Register => panic!("mma: a partition level walks Direct or Staged"),
            Tier::Memory => double_smem(lhs, rhs, self, space),
        }
    }
}

/// `Direct` at the register tier: the static walk over the whole level, every operand
/// window read at its use (the leaf loads transient fragments per contraction). No reuse —
/// that is `Staged`'s job — so a partition level wanting the staged register microkernel
/// declares [`Schedule::Staged`].
#[cube]
fn direct_register<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
) {
    let walk = comptime!(StaticWalk::over(&Space::merge(&[&lhs.space, &rhs.space])));
    #[unroll]
    for i in 0..comptime!(walk.total()) {
        let region = comptime!(walk.region(i));
        out.at_static(comptime!(region.clone())).mma(
            &lhs.at_static(comptime!(region.clone())),
            &rhs.at_static(region),
        );
    }
}

/// `Staged` at the memory tier: one [`Staging`] slot filled and consumed per region. The
/// slot owns the rendezvous (fill visible before any read, reads done before the refill);
/// `consume_final` every region, since no later fill publishes within an iteration.
#[cube]
fn staged_smem<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) {
    let mut slot = Staging::new(lhs, rhs);
    for region in Walk::over(space) {
        slot.fill(|s, pipe| {
            pipe.fill(&mut s.0, &lhs.at(&region));
            pipe.fill(&mut s.1, &rhs.at(&region));
        });
        slot.consume_final(|a, b| out.at(&region).mma(a, b));
    }
}

/// `Staged` at the register tier: per contraction step, stage each operand's fragments
/// ([`Tile::stage`] — where the legacy `A`-across-`n` / `B`-across-`m` reuse lives), then
/// walk the fragment grid contracting fragment triples through the ordinary recursion.
#[cube]
fn staged_register<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
) {
    // The MMA instruction shape, read off the final tiles.
    let fin = comptime!(out.space.final_space());
    let m = comptime!(fin.extent_at(fin.rank() - 2));
    let n = comptime!(fin.extent_at(fin.rank() - 1));
    let contracted = comptime!(contracted_axis(&lhs.space, &out.space));
    let k = comptime!(lhs.space.final_space().extent(contracted));
    let k_tiles = comptime!(lhs.space.count(contracted));

    // The fragment grid: the output's axes at this level. Trailing axes swapped so rows
    // run fastest — each staged `B` fragment feeds a consecutive burst of executes, the
    // legacy microkernel's emission order (worth ~1.3% on Metal).
    let grid = comptime!({
        let mut axes: Vec<Axis> = out.space.axes().collect();
        axes.swap(out.space.rank() - 2, out.space.rank() - 1);
        StaticWalk::over(&out.space.project(&axes))
    });

    // The contraction steps stay a *runtime* loop: `ki` only positions memory windows, and
    // unrolling it would keep every step's staged fragments live at once.
    for ki in 0..k_tiles {
        let a = CmmaPartition::stage(lhs, MatrixIdent::A, m, n, k, contracted, ki);
        let b = CmmaPartition::stage(rhs, MatrixIdent::B, m, n, k, contracted, ki);
        #[unroll]
        for i in 0..comptime!(grid.total()) {
            let region = comptime!(grid.region(i));
            out.at_static(comptime!(region.clone())).mma(
                &a.at_static(comptime!(region.clone())),
                &b.at_static(region),
            );
        }
    }
}

/// `DoubleBuffered` at the memory tier: two [`Staging`] slots, each a `(Tile<Lhs>, Tile<Rhs>)`
/// payload, driven `fill`/`consume` on alternating slots so one slot's fill overlaps the
/// other's compute. Each slot's synchronization is wrapped inside `fill`/`consume`, not here.
#[cube]
fn double_smem<Lhs: Numeric, Rhs: Numeric, Acc: Numeric>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
    space: Space,
) {
    // Each slot stages `lhs`/`rhs` into its own smem; the sync strategy and the allocation both live
    // in `Staging::new` (deduced from the operands' delivery), so the schedule stays out of it.
    let mut s0 = Staging::new(lhs, rhs);
    let mut s1 = Staging::new(lhs, rhs);

    // Double-buffering needs random access (prefetch the next region), so it indexes the `walk`
    // by hand rather than iterating.
    let walk = Walk::over(space);
    let n = walk.total();

    // prologue: prime slot 0 with region 0.
    let first = walk.region(0);
    s0.fill(|s, pipe| {
        pipe.fill(&mut s.0, &lhs.at(&first));
        pipe.fill(&mut s.1, &rhs.at(&first));
    });

    for p in 0..n / 2 {
        let even = p * 2;
        let odd = even + 1;

        // prefetch the odd region into slot 1 (its fill overlaps the compute below), then compute
        // the even region on slot 0.
        let odd_region = walk.region(odd);
        s1.fill(|s, pipe| {
            pipe.fill(&mut s.0, &lhs.at(&odd_region));
            pipe.fill(&mut s.1, &rhs.at(&odd_region));
        });
        let even_region = walk.region(even);
        s0.consume(|a, b| out.at(&even_region).mma(a, b));

        // prefetch the next even region back into slot 0 (if it exists), then compute the odd
        // region on slot 1. When a fill follows, its rendezvous publishes slot 1; when none does
        // (the walk's final region), `consume_final` publishes it first.
        let odd_region = walk.region(odd);
        if odd + 1 < n {
            let next_even = walk.region(odd + 1);
            s0.fill(|s, pipe| {
                pipe.fill(&mut s.0, &lhs.at(&next_even));
                pipe.fill(&mut s.1, &rhs.at(&next_even));
            });
            s1.consume(|a, b| out.at(&odd_region).mma(a, b));
        } else {
            s1.consume_final(|a, b| out.at(&odd_region).mma(a, b));
        }
    }

    // An odd total leaves the last region primed in slot 0 (by the final prefetch above, or the
    // prologue when `n == 1`) with no consumer in the loop; no fill follows, so `consume_final`
    // publishes it before draining.
    if n % 2 == 1 {
        let last = walk.region(n - 1);
        s0.consume_final(|a, b| out.at(&last).mma(a, b));
    }
}

/// The axis of `operand` the output drops — the contraction axis.
pub(crate) fn contracted_axis(operand: &Space, out: &Space) -> Axis {
    let contracted = operand.contracting(out);
    assert!(
        contracted.len() == 1,
        "cmma accumulator: the leaf contracts exactly one axis"
    );
    contracted[0]
}

/// The contraction depth `k`: the final-space extent of the contracted axis.
pub(crate) fn contracted_extent(operand: &Space, out: &Space) -> usize {
    operand.final_space().extent(contracted_axis(operand, out))
}
