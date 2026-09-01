//! The attention verb's leaves, one per fold shape:
//!
//! * [`score`](crate::Tile::score) and [`mix`](crate::Tile::mix): the shared-memory fold's two
//!   matmuls (the score into a materialized tile, the value mix with the online-softmax rescale
//!   fused in). Prefill's shape: real query blocks, a score tile between two matmuls.
//! * [`stream`]: the register-resident fold. No score tile and no barriers; each position's dot
//!   closes with a plane sum and feeds the accumulator immediately. Decode's shape: a single query
//!   position per group.
//!
//! Both matmuls contract through the instruction their accumulator's space states, and nothing
//! else picks it: the software instruction runs [`columns`], where a unit owns every
//! `CUBE_DIM_X`-th column, and a hardware one runs [`fragments`], where a plane owns every
//! `planes`-th fragment. Like [`softmax`](crate::Tile::softmax) these are leaf ops on
//! shared-memory tiles: the caller owns the walk and the syncs.

mod columns;
mod fragments;
mod stream;

pub use stream::*;
// columns and fragments add `Tile` impls only; nothing to re-export.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// The score matmul: `self[r, c] = dot(q[r, :], k[c, :])`.
    ///
    /// `self` is a rank-2 `{rows, cols}` scalar score tile, `q` the `{rows, head_dim}` queries and
    /// `k` one block of `{cols, head_dim}` keys, its trailing axis the contracted head dim. Columns
    /// at or past `cols_bound` are not read: `k` may end before the block does, and the softmax's
    /// mask probe overwrites those cells anyway. The hardware arm skips them by the whole
    /// fragment, so one straddling the bound is still read to its edge. The caller syncs after.
    ///
    /// The instruction states its own shape, as it does everywhere else: the software one carries
    /// a register budget, which caps the rows a visit keeps live, and the hardware one carries
    /// none, so its fragment is the box the level it sits on cuts.
    pub fn score<EI: Numeric>(&mut self, q: &Tile<EI>, k: &Tile<EI>, cols_bound: usize) {
        match comptime!(instruction(&self.space, "score")) {
            Instruction::Registers { config } => {
                let width = q.vector_size();
                let rows = comptime!(rows_per_visit(config, width, &self.space));
                self.score_columns(q, k, cols_bound, rows)
            }
            Instruction::Cmma => self.score_fragments(q, k, cols_bound),
            Instruction::Mma { .. } => comptime!(panic!(
                "score: the manual-mma form reads its operands row-major, so it cannot read the \
                 keys as the transposed matrix this contraction needs; state Instruction::Cmma or \
                 Instruction::registers(budget)"
            )),
        }
    }

    /// The value matmul with the online-softmax rescale fused in:
    /// `self[r, :] = self[r, :] · factors[r] + Σ_{c < cols_bound} p[r, c] · val[c, :]`.
    ///
    /// `self` is a rank-2 `{rows, val_dim}` accumulator, `p` the `{rows, cols}` probabilities and
    /// `val` one block of `{cols, val_dim}` values. Columns at or past `cols_bound` are skipped:
    /// stale cache beyond the attended prefix (possibly NaN) must not ride a zero probability. The
    /// hardware arm skips them by the whole contraction step, so one straddling the bound is still
    /// read to its edge. The caller syncs on both sides.
    ///
    /// The instruction states its shape as it does for [`score`](Tile::score).
    pub fn mix<EP: Numeric, EI: Numeric>(
        &mut self,
        p: &Tile<EP>,
        val: &Tile<EI>,
        factors: &Tile<EA>,
        cols_bound: usize,
    ) {
        match comptime!(instruction(&self.space, "mix")) {
            Instruction::Registers { config } => {
                let width = val.vector_size();
                let rows = comptime!(rows_per_visit(config, width, &self.space));
                self.mix_columns(p, val, factors, cols_bound, rows)
            }
            Instruction::Cmma => self.mix_fragments(p, val, factors, cols_bound),
            Instruction::Mma { .. } => comptime!(panic!(
                "mix: the manual-mma form is not wired for the attention leaves; state \
                 Instruction::Cmma or Instruction::registers(budget)"
            )),
        }
    }
}

/// The instruction this accumulation contracts through, stated by the level its space was cut at
/// ([`Tiling::instruction`]). A leaf finding none is a plan that never stated one, and says so
/// while the kernel expands rather than picking an arm on its own.
fn instruction(space: &Space, verb: &str) -> Instruction {
    space.instruction().unwrap_or_else(|| {
        panic!(
            "{verb}: this accumulator's space states no instruction; end its tiling with \
             `.instruction(Instruction::Cmma, |l| ...)`, or `Instruction::registers(budget)` for \
             the software one, so the level it is cut at says what runs on its cells"
        )
    })
}

/// The `rows × cols` box one visit covers: the edges of the level the space's instruction sits on,
/// or the whole tile where that level cut nothing. Every axis above the innermost multiplies into
/// the row edge, so a tile carrying its rows as several axes reads like the flat one it is laid
/// out as.
pub(super) fn visit_box(space: &Space) -> (usize, usize) {
    let sub = space.sub_tile_space();
    let rank = sub.rank();
    (
        (0..rank - 1).map(|p| sub.extent_at(p)).product(),
        sub.extent_at(rank - 1),
    )
}

/// How many rows one visit of the software instruction keeps live, out of its
/// [`budget`](RegisterBlock::budget) of accumulator registers: the visit holds one `width`-wide
/// line per row, so a wider line buys fewer rows, not more registers.
///
/// The largest such count that also *divides* the accumulator, so every visit is the same shape
/// and none straddles the last row: a visit is what a worker picks up, and a ragged one has no
/// comptime height to unroll over. Never zero, since a visit holding no row contracts nothing.
fn rows_per_visit(config: RegisterBlock, width: usize, space: &Space) -> usize {
    let rows = space.extent_at(space.rank() - 2);
    let cap = (config.budget / width).max(1).min(rows);
    (1..=cap)
        .rev()
        .find(|c| rows.is_multiple_of(*c))
        .unwrap_or(1)
}
