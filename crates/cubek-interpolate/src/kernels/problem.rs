//! The problem a routine computes.

use crate::definition::{
    InterpolateCost, InterpolateForwardProblem, InterpolateProblem, get_transform, mode_properties,
};
use crate::kernels::coordinate::Rational;
use crate::kernels::geometry::TileGeometry;
use crate::kernels::space::{CHANNEL, InterpolateSpace};
use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubek_microkernel::interpolate::InterpolateOperand;
use cubek_std::Hardware;
use cubek_std::routine::EstimatedProblem;

/// An NHWC input resampled to the output's height and width, as the host describes it.
#[derive(Debug, Clone)]
pub struct InterpolateRoutineProblem {
    pub forward: InterpolateForwardProblem,
    /// The element both tensors are stored in.
    pub element: ElemType,
    /// The alignment of the input's and the output's buffers in elements.
    pub elem_alignments: [ElemAlignment; 2],
}

impl InterpolateRoutineProblem {
    /// The taps along each resampled axis of one output position.
    pub fn taps(&self) -> usize {
        mode_properties(self.forward.options.mode).taps
    }

    /// The taps on either side of the one an output position lands nearest.
    pub fn radius(&self) -> usize {
        (self.taps() - 1) / 2
    }

    /// Where an output row lands among the input rows.
    pub fn row(&self) -> Rational {
        let forward = &self.forward;
        Rational::of(get_transform(
            forward.input_height,
            forward.output_height,
            forward.options,
        ))
    }

    /// Where an output column lands among the input columns.
    pub fn col(&self) -> Rational {
        let forward = &self.forward;
        Rational::of(get_transform(
            forward.input_width,
            forward.output_width,
            forward.options,
        ))
    }

    /// The space `geometry` cuts on planes of `plane_width` units.
    pub fn space(&self, geometry: TileGeometry, plane_width: usize) -> InterpolateSpace {
        let forward = &self.forward;
        InterpolateSpace {
            batch: forward.batch,
            height: forward.output_height,
            width: forward.output_width,
            channels: forward.channels,
            plane_size: plane_width,
            taps: self.taps(),
            geometry,
        }
    }

    /// Whether every tap of every output position reads inside the input, so the gather's reads
    /// go unchecked.
    pub fn taps_in_bounds(&self) -> bool {
        let forward = &self.forward;
        let (taps, radius) = (self.taps(), self.radius());
        tap_range_in_bounds(
            self.row(),
            forward.output_height,
            forward.input_height,
            taps,
            radius,
        ) && tap_range_in_bounds(
            self.col(),
            forward.output_width,
            forward.input_width,
            taps,
            radius,
        )
    }

    /// The width both operands bind at on `hw`: to the requirements the routine enqueued in
    /// `vectorization`, the binding adds each buffer's alignment, and a single value where the
    /// channel axis overhangs, whose masked access counts in lines and would clip wrongly.
    pub fn width(
        &self,
        hw: &Hardware,
        partitioning: &Partitioning,
        mut vectorization: Vectorization<InterpolateOperand>,
    ) -> usize {
        for (operand, alignment) in [InterpolateOperand::Input, InterpolateOperand::Output]
            .into_iter()
            .zip(self.elem_alignments)
        {
            vectorization.require(VectorRequirement::Aligned { operand, alignment });
            if partitioning.overhangs(CHANNEL) {
                vectorization.require(VectorRequirement::Elementwise { operand });
            }
        }
        vectorization.width(InterpolateOperand::Input, |_| {
            hw.vector_ceiling(self.element)
        })
    }
}

impl EstimatedProblem for InterpolateRoutineProblem {
    fn demand(&self) -> Work {
        InterpolateCost::new(InterpolateProblem::Forward(self.forward.clone()), self.element)
            .work()
    }
}

fn tap_range_in_bounds(
    source: Rational,
    outputs: usize,
    input: usize,
    taps: usize,
    radius: usize,
) -> bool {
    let first = source.offset.div_euclid(source.divisor as isize) - radius as isize;
    let numerator = (outputs.saturating_sub(1) as isize) * source.scale as isize + source.offset;
    let last = numerator.div_euclid(source.divisor as isize) + (taps - 1 - radius) as isize;
    first >= 0 && last < input as isize
}
