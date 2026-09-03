//! The two matmuls of the shared-memory attention fold, and the one statement that decides how
//! each contracts: the [`Instruction`] its accumulator's space carries.

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
        match comptime!(self.space.instruction().expect(
            "score: this accumulator's space states no instruction; end its tiling with \
             `.instruction(Instruction::Cmma, |l, o| ...)`, or `Instruction::registers(budget)` for \
             the software one, so the level it is cut at says what runs on its cells"
        )) {
            Instruction::Registers { config } => self.score_columns(q, k, cols_bound, config),
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
        match comptime!(self.space.instruction().expect(
            "mix: this accumulator's space states no instruction; end its tiling with \
             `.instruction(Instruction::Cmma, |l, o| ...)`, or `Instruction::registers(budget)` for \
             the software one, so the level it is cut at says what runs on its cells"
        )) {
            Instruction::Registers { config } => {
                self.mix_columns(p, val, factors, cols_bound, config)
            }
            Instruction::Cmma => self.mix_fragments(p, val, factors, cols_bound),
            Instruction::Mma { .. } => comptime!(panic!(
                "mix: the manual-mma form is not wired for the attention leaves; state \
                 Instruction::Cmma or Instruction::registers(budget)"
            )),
        }
    }
}
