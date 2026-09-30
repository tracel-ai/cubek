//! The online softmax as a tile op: a score folded along one axis, one block of it at a time,
//! into each slice's running max and sum, where its holder already keeps the score.

use cubecl::prelude::*;

use crate::*;

/// The running max `m` and sum `l` of every slice along `axis` of a score, in the registers of
/// its holder.
///
/// The score is cut where its holder keeps it ([`AxisSlices`]): a unit's register block holds
/// whole slices of its own, and a plane's window of shared memory is the plane's, every unit of
/// it keeping the state of every slice.
#[derive(CubeType)]
pub struct OnlineSoftmax<E: Float> {
    m: Array<E>,
    l: Array<E>,
    #[cube(comptime)]
    slices: usize,
    #[cube(comptime)]
    axis: Axis,
}

#[cube]
impl<E: Float> OnlineSoftmax<E> {
    /// The state of every slice along `axis` of a score over the same box as `score`, before
    /// any block is folded.
    pub fn along<S: Numeric>(score: &Tile<S>, #[comptime] axis: Axis) -> OnlineSoftmax<E> {
        OnlineSoftmax::<E>::new(comptime!(score.place.space.slices_along(axis)), axis)
    }

    /// The state of `slices` slices along `axis` before any block is folded: what a holder opens
    /// before the walk that hands it its score, as a cube's plane does before the cube's ring.
    pub fn new(#[comptime] slices: usize, #[comptime] axis: Axis) -> OnlineSoftmax<E> {
        let mut m = Array::<E>::new(slices);
        let mut l = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            m[i] = E::min_value();
            l[i] = E::from_int(0);
        }
        OnlineSoftmax::<E> { m, l, slices, axis }
    }

    /// Fold one block of `score` into the slices. The block becomes `score · scale + bias`, the
    /// slices' max and sum move on, and `score` is left holding `exp(score − max)`: the block's
    /// probabilities, unnormalized. Returns each slice's `exp(max before − max after)`, what
    /// anything summed against the earlier blocks is multiplied by ([`AxisSlices::mul`]).
    ///
    /// `bias` is a procedural tile windowed to the same region as `score`. A masked cell's bias is
    /// `E::min_value()`; a slice whose every cell so far is masked holds zeros and a zero sum. On a
    /// plane's window every unit of the plane calls it, and the caller meets the plane before it
    /// and before the probabilities are read.
    pub fn step(&mut self, score: &Tile<E>, bias: &Tile<E>, scale: E) -> Array<E> {
        let mut slices = score.along(self.axis);
        slices.scale_add(scale, bias);
        let max = slices.maxima(&self.m);
        slices.exp_minus(&max);
        let sum = slices.sums();
        let mut correction = Array::<E>::new(self.slices);
        #[unroll]
        for i in 0..self.slices {
            correction[i] = (self.m[i] - max[i]).exp();
            self.l[i] = correction[i] * self.l[i] + sum[i];
            self.m[i] = max[i];
        }
        correction
    }

    /// Each slice's `1 / l`, what its sum of probabilities times values is normalized by; zero
    /// for a slice whose every cell was masked.
    pub fn recip_l(&self) -> Array<E> {
        let mut recip = Array::<E>::new(self.slices);
        #[unroll]
        for i in 0..self.slices {
            let live = self.l[i] > E::from_int(0);
            recip[i] = select(live, self.l[i].recip(), E::from_int(0));
        }
        recip
    }
}
