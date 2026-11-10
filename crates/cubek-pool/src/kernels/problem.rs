//! The problem a routine computes.

use crate::kernels::geometry::TileGeometry;
use crate::kernels::space::{CHANNEL, PoolSpace, WindowAxis};
use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubek_microkernel::pool::PoolOperand;
use cubek_std::Hardware;
use cubek_std::routine::EstimatedProblem;

/// An NHWC input max pooled to the output's height and width, as the host describes it.
#[derive(Debug, Clone)]
pub struct PoolRoutineProblem {
    pub batch: usize,
    pub input: [usize; 2],
    pub output: [usize; 2],
    pub channels: usize,
    /// The window along the rows, then along the columns.
    pub windows: [WindowAxis; 2],
    /// The element both tensors are stored in.
    pub element: ElemType,
    /// The alignment of the input's and the output's buffers in elements.
    pub elem_alignments: [ElemAlignment; 2],
}

impl PoolRoutineProblem {
    /// The space `geometry` cuts on planes of `plane_width` units.
    pub fn space(&self, geometry: TileGeometry, plane_width: usize) -> PoolSpace {
        PoolSpace {
            batch: self.batch,
            height: self.output[0],
            width: self.output[1],
            channels: self.channels,
            plane_size: plane_width,
            taps: self.windows.map(|window| window.taps),
            geometry,
        }
    }

    /// Whether every tap of every output position reads inside the input, so the gather's reads
    /// go unchecked.
    pub fn taps_in_bounds(&self) -> bool {
        (0..2).all(|axis| self.windows[axis].in_bounds(self.output[axis], self.input[axis]))
    }

    /// The width both operands bind at on `hw`: to the requirements the routine enqueued in
    /// `vectorization`, the binding adds each buffer's alignment.
    pub fn width(
        &self,
        hw: &Hardware,
        partitioning: &Partitioning,
        mut vectorization: Vectorization<PoolOperand>,
    ) -> usize {
        for (operand, alignment) in [PoolOperand::Input, PoolOperand::Output]
            .into_iter()
            .zip(self.elem_alignments)
        {
            vectorization.require(VectorRequirement::Aligned { operand, alignment });
            if partitioning.overhangs(CHANNEL) {
                vectorization.require(VectorRequirement::Elementwise { operand });
            }
        }
        vectorization.width(PoolOperand::Input, |_| hw.vector_ceiling(self.element))
    }
}

impl EstimatedProblem for PoolRoutineProblem {
    /// One comparison a tap, every input read once and every output written once.
    fn demand(&self) -> Work {
        let outputs = self.batch * self.output[0] * self.output[1] * self.channels;
        let taps = self.windows[0].taps * self.windows[1].taps;
        let inputs = self.batch * self.input[0] * self.input[1] * self.channels;
        Work {
            compute_ops: outputs * taps,
            bytes: (inputs + outputs) * self.element.size(),
        }
    }
}
