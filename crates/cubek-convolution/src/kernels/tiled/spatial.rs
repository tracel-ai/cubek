use crate::launch::SpatialWindow;
use cubecl::tile::Dimension;
use cubecl::tile::layout::{PhysicalAxisMap, Scale};
use serde::{Deserialize, Serialize};

/// One spatial dim of a convolution: the extents it reads and writes along it and the window
/// between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialDim {
    /// The extent of the gathered operand: the input, or the output gradient in the data backward.
    pub gathered: usize,
    pub kernel: usize,
    /// The output's extent.
    pub out: usize,
    pub window: SpatialWindow,
}

impl SpatialDim {
    /// Whether every position the window reads lies in the gathered operand, so no read needs a
    /// check of its own.
    pub fn reads_inside(&self) -> bool {
        let last = self.out.saturating_sub(1) * self.window.stride
            + self.kernel.saturating_sub(1) * self.window.dilation;
        self.window.padding == 0 && last < self.gathered
    }
}

impl SpatialDim {
    /// The cells before the input gradient's start that the phases reach, never zero where the
    /// extent overruns a whole number of strides, so the output map keeps its edge checks.
    pub fn phase_lead(&self) -> usize {
        match (self.window.padding, self.out % self.window.stride) {
            (0, 0) => 0,
            (0, _) => self.window.stride,
            (padding, _) => padding,
        }
    }

    /// The cells of one phase of the input gradient along this dim: the positions
    /// `stride * cell + phase - lead`.
    pub fn phased_out(&self) -> usize {
        (self.out + self.phase_lead()).div_ceil(self.window.stride)
    }

    /// The taps of one phase of the weight along this dim: the taps `stride * tap + phase`.
    pub fn phased_taps(&self) -> usize {
        self.kernel.div_ceil(self.window.stride)
    }

    /// Whether some phase has a tap past the weight's end, which must read as zero.
    pub fn has_ragged_taps(&self) -> bool {
        !self.kernel.is_multiple_of(self.window.stride)
    }

    /// The output gradient's axis over `out` and `tap` that a phase reads: `out - tap` less the
    /// stride's worth of rows the lead takes beyond the padding. Negative before the gradient's
    /// start, and past its end where the phase has more cells than the gradient has rows.
    pub fn phased_gradient_map(&self, out: Dimension, tap: Dimension) -> PhysicalAxisMap {
        let behind = (self.phase_lead() - self.window.padding) / self.window.stride;
        PhysicalAxisMap::scaled(&[(out, Scale::Static(1)), (tap, Scale::Static(-1))])
            .shifted(-(behind as isize))
    }

    /// The input gradient's axis over `out` and `phase`: `stride * out + phase - lead`.
    pub fn phased_out_map(&self, out: Dimension, phase: Dimension) -> PhysicalAxisMap {
        PhysicalAxisMap::disjoint(&[(out, self.window.stride), (phase, 1)])
            .shifted(-(self.phase_lead() as isize))
    }

    /// The weight's tap axis over `tap` and `phase`: `stride * tap + phase`. Where some phase has a
    /// tap past the end the stride is a launch value ([`ConvProblem::phased_tap_strides`]), so the map is
    /// not a radix chain that the tile would take as in bounds.
    pub fn phased_tap_map(&self, tap: Dimension, phase: Dimension) -> PhysicalAxisMap {
        match self.has_ragged_taps() {
            true => PhysicalAxisMap::scaled(&[
                (
                    tap,
                    Scale::Dynamic {
                        max: self.window.stride,
                    },
                ),
                (phase, Scale::Static(1)),
            ]),
            false => PhysicalAxisMap::disjoint(&[(tap, self.window.stride), (phase, 1)]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::ConvDimension;
    use crate::launch::{SpatialExtent, TapCount};
    use cubecl::tile::layout::Offset;

    fn strided(kernel: usize, stride: usize, padding: usize, out: usize) -> SpatialDim {
        let window = SpatialWindow {
            stride,
            padding,
            dilation: 1,
        };
        let gathered = window.output(SpatialExtent(out), TapCount(kernel));
        SpatialDim {
            gathered,
            kernel,
            out,
            window,
        }
    }

    /// The dims the phase tests sweep: every window whose forward output has at least one cell.
    fn swept_dims() -> impl Iterator<Item = SpatialDim> {
        (1..=7_usize).flat_map(|kernel| {
            (2..=4_usize).flat_map(move |stride| {
                (0..kernel).flat_map(move |padding| {
                    (1..=17_usize)
                        .filter(move |out| out + 2 * padding >= kernel)
                        .map(move |out| strided(kernel, stride, padding, out))
                })
            })
        })
    }

    /// The input gradient cells that `phase` of `dim` writes, as `(cell, position)`.
    fn written_cells(dim: &SpatialDim, phase: usize) -> impl Iterator<Item = (usize, usize)> {
        let (stride, lead, out) = (dim.window.stride, dim.phase_lead(), dim.out);
        (0..dim.phased_out())
            .map(move |cell| (cell, (stride * cell + phase).wrapping_sub(lead)))
            .filter(move |&(_, position)| position < out)
    }

    /// The phases cover every input gradient cell once.
    #[test]
    fn phases_partition_the_input_gradient() {
        for dim in swept_dims() {
            let mut written = vec![0; dim.out];
            for phase in 0..dim.window.stride {
                for (_, position) in written_cells(&dim, phase) {
                    written[position] += 1;
                }
            }
            assert!(written.iter().all(|&count| count == 1), "{dim:?}");
        }
    }

    /// A phase's cell and tap read the gradient row that the forward window paired with that cell
    /// and weight tap, or nothing where the forward window has no such pair.
    #[test]
    fn each_tap_pairs_the_forward_row() {
        for dim in swept_dims() {
            let (stride, padding) = (dim.window.stride as isize, dim.window.padding as isize);
            let rows = dim.gathered as isize;
            let behind = ((dim.phase_lead() - dim.window.padding) / dim.window.stride) as isize;
            for phase in 0..dim.window.stride {
                for (cell, position) in written_cells(&dim, phase) {
                    for tap in 0..dim.phased_taps() {
                        let weight = dim.window.stride * tap + phase;
                        let row = cell as isize - tap as isize - behind;
                        let reads = weight < dim.kernel && (0..rows).contains(&row);
                        let paired = position as isize + padding - weight as isize;
                        let pair = (paired.rem_euclid(stride) == 0)
                            .then_some(paired.div_euclid(stride))
                            .filter(|o| weight < dim.kernel && (0..rows).contains(o));
                        assert_eq!(reads.then_some(row), pair, "{dim:?} x{position} k{weight}");
                    }
                }
            }
        }
    }

    /// A map that is a radix chain over its extents reads as in bounds, so a dim whose cells run
    /// past the buffer's end must not have one.
    #[test]
    fn a_phased_dim_that_overruns_its_axis_has_a_map_that_is_not_a_radix_chain() {
        let chain = |dim: SpatialDim| dim.phased_out() * dim.window.stride == dim.out;
        let exact = strided(4, 2, 0, 8);
        assert_eq!(exact.phase_lead(), 0);
        assert!(chain(exact) && !exact.has_ragged_taps());
        let (out, phase) = (ConvDimension::Out(0), ConvDimension::Phase(0));
        let map = |dim: SpatialDim| dim.phased_out_map(out.dimension(), phase.dimension());
        assert_eq!(map(exact).offset(), Offset::Static(0));
        assert_ne!(map(strided(3, 2, 0, 7)).offset(), Offset::Static(0));
        let ragged = strided(3, 2, 0, 7);
        assert_eq!(ragged.phase_lead(), 2);
        assert!(ragged.has_ragged_taps());
        assert_eq!(strided(3, 2, 1, 7).phase_lead(), 1);
    }
}
