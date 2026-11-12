use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{B, ConvProblem};
use crate::launch::ConvolutionArgs;
use cubecl::ir::ElemType;
use cubecl::throughput::{MemoryAccess, compute_throughput_key};
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKey};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::autotune::Ceiling;
use cubek_std::{Hardware, Instructions, manual_mma};
use serde::{Deserialize, Serialize};

/// What makes two convolutions want the same setting.
///
/// Every extent is exact, since a model has few layer shapes and anchoring would merge layers that
/// want different settings, except the batch: a caller varies it freely, so it is anchored to the
/// power of two above it. The widths are how wide each operand's alignment, and its view's start,
/// lets it be read, the output's excepted: the cube staged routine writes it one element at a time.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ConvTuneKey {
    operation: ConvolutionOperation,
    lhs: ElemType,
    rhs: ElemType,
    out: ElemType,
    sum: ElemType,
    batch: usize,
    gathered: [usize; 2],
    out_extent: [usize; 2],
    kernel: [usize; 2],
    channels_in: usize,
    channels_out: usize,
    window: ConvolutionArgs<2>,
    lhs_width: usize,
    rhs_width: usize,
}

impl ConvTuneKey {
    /// The widest width a key tells apart is two to this power: a wider alignment shares the key.
    const MOST_WIDTH_POWER: u32 = 5;

    /// The batch `problem` is keyed by: the power of two above it, unless a launch of that many
    /// batches would put more cubes along the grid's third dimension (the batches beside the
    /// positions across) than the device spawns, which settings may then disagree on within one
    /// bucket.
    fn batch(problem: &ConvProblem, hw: &Hardware) -> usize {
        let roles = problem.roles();
        let bucket = problem.batch.next_power_of_two();
        let rides_grid = roles.free.batch == B;
        let cubes = bucket.saturating_mul(problem.extent(roles.free.across));
        let most_cubes = hw.properties.hardware.max_cube_count.2 as usize;
        match rides_grid && cubes > most_cubes {
            true => problem.batch,
            false => bucket,
        }
    }

    /// The widest power-of-two width `alignment` admits for a view starting at `start`.
    fn width(alignment: ElemAlignment, start: ElemAlignment) -> usize {
        (0..=Self::MOST_WIDTH_POWER)
            .map(|power| 1usize << power)
            .take_while(|&width| alignment.admits(width) && start.admits(width))
            .last()
            .unwrap_or(1)
    }

    /// The alignment a view of `problem`'s operand starts at, as the launch it prepares reads it.
    fn start(problem: &ConvProblem, operand: MatmulOperand) -> ElemAlignment {
        ElemAlignment::at_start(
            problem.start_bytes.get(operand),
            problem.elements.get(operand).size(),
        )
    }

    /// The bound the convolution has: the largest instruction the device offers at the key's
    /// elements accumulating in the key's sum (plain arithmetic where it offers none), or the read
    /// bus, whichever is higher for the key.
    pub(super) fn ceiling(&self, hw: &Hardware) -> Ceiling {
        let instruction = hw
            .properties
            .features
            .matmul
            .cmma
            .iter()
            .copied()
            .chain(manual_mma(hw))
            .for_triple(self.lhs, self.rhs, self.sum)
            .largest()
            .map(|shape| (shape.m, shape.n, shape.k));
        Ceiling::Either {
            compute: compute_throughput_key(instruction, self.lhs, self.sum),
            memory: MemoryAccess::Read,
        }
    }
}

impl ConvTuneKey {
    /// The key of `problem` on `hw`.
    pub(super) fn of(problem: &ConvProblem, hw: &Hardware) -> Self {
        let width = |operand| {
            Self::width(
                problem.alignments.get(operand),
                Self::start(problem, operand),
            )
        };
        Self {
            operation: problem.operation,
            lhs: problem.elements.lhs,
            rhs: problem.elements.rhs,
            out: problem.elements.out,
            sum: problem.sum,
            batch: Self::batch(problem, hw),
            gathered: problem.gathered,
            out_extent: problem.out,
            kernel: problem.kernel,
            channels_in: problem.channels_in,
            channels_out: problem.channels_out,
            window: problem.window,
            lhs_width: width(MatmulOperand::Lhs),
            rhs_width: width(MatmulOperand::Rhs),
        }
    }
}

/// The problem a key describes: dense operands of the key's elements at its widths, starting at
/// their buffers' starts.
impl From<&ConvTuneKey> for ConvProblem {
    fn from(key: &ConvTuneKey) -> Self {
        let alignments = MatmulOperands {
            lhs: ElemAlignment::dense_rows(key.lhs_width),
            rhs: ElemAlignment::dense_rows(key.rhs_width),
            out: ElemAlignment::dense_rows(1),
        };
        Self {
            operation: key.operation,
            batch: key.batch,
            channels_in: key.channels_in,
            gathered: key.gathered,
            channels_out: key.channels_out,
            kernel: key.kernel,
            out: key.out_extent,
            window: key.window,
            elements: MatmulOperands {
                lhs: key.lhs,
                rhs: key.rhs,
                out: key.out,
            },
            sum: key.sum,
            alignments,
            start_bytes: MatmulOperands::all(0),
        }
    }
}
