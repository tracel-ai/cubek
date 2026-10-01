//! The online softmax as a tile op: a score taken in along one axis, one block of it at a time,
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
    /// any block is taken in.
    pub fn along<S: Numeric>(score: &Tile<S>, #[comptime] axis: Axis) -> OnlineSoftmax<E> {
        OnlineSoftmax::<E>::new(comptime!(score.place.space.slices_along(axis)), axis)
    }

    /// The state of `slices` slices along `axis` before any block is taken in: what a holder opens
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

    /// Take one block of `score` into the slices. The block becomes `score · scale + bias`, the
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

    /// What this plane's sum against the values is multiplied by once its last block is taken in:
    /// each slice's `1 / l`, or, where the cube's planes split this axis, each holding part of the
    /// same slices, the correction to the merged max times `1 / L` of the merged sum, so the
    /// planes' products only remain to be added ([`PlanesOutput`]). `plane` is this plane's
    /// region of the cube; the partitioning it was cut by says how many planes share its slices.
    ///
    /// Where planes split the axis it is a meeting of the cube: every unit of every plane calls it,
    /// each holding its plane's state as a plane's window or fragments keep it, and it meets the
    /// cube three times through shared memory. The planes' values are read in plane order, so the
    /// merged sum is the same bits from run to run. A slice every plane masked stays at zero.
    pub fn normalizer(&mut self, plane: &Region) -> Array<E> {
        let axis = self.axis;
        let planes = comptime!(
            plane
                .path
                .planes_along(|along| along == axis)
                .iter()
                .map(|&(_, count)| count)
                .product::<usize>()
        );
        let mut factor = Array::<E>::new(self.slices);
        if comptime!(planes == 1) {
            factor = self.recip_l();
        } else {
            let correction = self.merge_planes(plane.coord(axis), planes);
            let recip = self.recip_l();
            #[unroll]
            for i in 0..self.slices {
                factor[i] = correction[i] * recip[i];
            }
        }
        factor
    }

    /// Merge this state with those of the `planes` planes, this one at `plane`, that took in other
    /// parts of the same slices: every plane leaves holding the merged max and sum. Returns each
    /// slice's `exp(max before − max after)`. The first unit of each plane writes, and the cube
    /// meets before each read and before the values read are overwritten.
    fn merge_planes(&mut self, plane: usize, #[comptime] planes: usize) -> Array<E> {
        let slices = self.slices;
        let mut shared = Shared::<[E]>::new_slice(comptime!(planes * slices));
        let writes = UNIT_POS_PLANE == 0;
        let own = plane * slices;

        if writes {
            #[unroll]
            for i in 0..slices {
                shared[own + i] = self.m[i];
            }
        }
        sync_cube();
        let mut max = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            max[i] = E::min_value();
            #[unroll]
            for p in 0..planes {
                max[i] = max[i].max(shared[p * slices + i]);
            }
        }
        // Every plane has read the maxima before any overwrites them with its sum.
        sync_cube();

        let mut correction = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            correction[i] = (self.m[i] - max[i]).exp();
        }
        if writes {
            #[unroll]
            for i in 0..slices {
                shared[own + i] = correction[i] * self.l[i];
            }
        }
        sync_cube();
        #[unroll]
        for i in 0..slices {
            let mut sum = E::from_int(0);
            #[unroll]
            for p in 0..planes {
                sum += shared[p * slices + i];
            }
            self.l[i] = sum;
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
