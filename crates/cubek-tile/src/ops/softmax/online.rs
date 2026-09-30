//! The online softmax as a tile op: a score folded into its rows' running max and sum one block
//! of its reduced axis at a time, where its holder already keeps the score.

use cubecl::prelude::*;

use crate::*;

/// The running max `m` and sum `l` of every row of a score its holder owns, in registers.
///
/// The score's rows are read where its holder keeps them ([`Rows`]): a unit's register block holds
/// whole rows of its own, and a plane's window of shared memory is the plane's, every unit of it
/// keeping the state of every row.
#[derive(CubeType)]
pub struct OnlineSoftmax<E: Float> {
    m: Array<E>,
    l: Array<E>,
    #[cube(comptime)]
    rows: usize,
}

#[cube]
impl<E: Float> OnlineSoftmax<E> {
    /// The state of every row of `score` before any block is folded.
    pub fn over<S: Numeric>(score: &Tile<S>) -> OnlineSoftmax<E> {
        OnlineSoftmax::<E>::new(comptime!(score.place.space.rows()))
    }

    /// The state of `rows` rows before any block is folded: what a holder opens before the walk
    /// that hands it its score, as a cube's plane does before the cube's ring.
    pub fn new(#[comptime] rows: usize) -> OnlineSoftmax<E> {
        let mut m = Array::<E>::new(rows);
        let mut l = Array::<E>::new(rows);
        #[unroll]
        for r in 0..rows {
            m[r] = E::min_value();
            l[r] = E::from_int(0);
        }
        OnlineSoftmax::<E> { m, l, rows }
    }

    /// Fold one block of `score` into the rows. The block becomes `score · scale + bias`, the rows'
    /// max and sum move on, and `score` is left holding `exp(score − max)`: the block's
    /// probabilities, unnormalized. Returns each row's `exp(max before − max after)`, what anything
    /// summed against the earlier blocks is multiplied by ([`Rows::mul`]).
    ///
    /// `bias` is a procedural tile windowed to the same region as `score`. A masked cell's bias is
    /// `E::min_value()`; a row whose every cell so far is masked holds zeros and a zero sum. On a
    /// plane's window every unit of the plane calls it, and the caller meets the plane before it
    /// and before the probabilities are read.
    pub fn step(&mut self, score: &Tile<E>, bias: &Tile<E>, scale: E) -> Array<E> {
        let mut rows = score.rows();
        rows.scale_add(scale, bias);
        let max = rows.maxima(&self.m);
        rows.exp_minus(&max);
        let sum = rows.sums();
        let mut correction = Array::<E>::new(self.rows);
        #[unroll]
        for r in 0..self.rows {
            correction[r] = (self.m[r] - max[r]).exp();
            self.l[r] = correction[r] * self.l[r] + sum[r];
            self.m[r] = max[r];
        }
        correction
    }

    /// Each row's `1 / l`, what its sum of probabilities times values is normalized by; zero for
    /// a row whose every cell was masked.
    pub fn recip_l(&self) -> Array<E> {
        let mut recip = Array::<E>::new(self.rows);
        #[unroll]
        for r in 0..self.rows {
            let live = self.l[r] > E::from_int(0);
            recip[r] = select(live, self.l[r].recip(), E::from_int(0));
        }
        recip
    }
}
