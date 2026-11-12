use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{B, ConvDim, ConvProblem};
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
pub struct CubeStagedConvTuneKey {
    pub operation: ConvolutionOperation,
    pub lhs: ElemType,
    pub rhs: ElemType,
    pub out: ElemType,
    pub sum: ElemType,
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub spatial: Vec<ConvDim>,
    pub lhs_width: usize,
    pub rhs_width: usize,
}

impl CubeStagedConvTuneKey {
    /// The widest width a key tells apart is two to this power: a wider alignment shares the key.
    const MOST_WIDTH_POWER: u32 = 5;

    /// The batch `problem` is keyed by: the power of two above it, unless a launch of that many
    /// batches would put more cubes along the grid's third dimension (the batches beside the
    /// positions across) than the device spawns, which settings may then disagree on within one
    /// bucket.
    fn batch(problem: &ConvProblem, hw: &Hardware) -> usize {
        let free = problem.roles().free;
        let bucket = problem.batch.next_power_of_two();
        let grid = || free.outer.iter().chain([&free.across]);
        let rides_grid = grid().any(|&axis| axis == B);
        let cubes = grid()
            .map(|&axis| match axis == B {
                true => bucket,
                false => problem.extent(axis),
            })
            .fold(1, usize::saturating_mul);
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

impl CubeStagedConvTuneKey {
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
            channels_in: problem.channels_in,
            channels_out: problem.channels_out,
            spatial: problem.spatial.clone(),
            lhs_width: width(MatmulOperand::Lhs),
            rhs_width: width(MatmulOperand::Rhs),
        }
    }
}

/// The problem a key describes: dense operands of the key's elements at its widths, starting at
/// their buffers' starts.
impl From<&CubeStagedConvTuneKey> for ConvProblem {
    fn from(key: &CubeStagedConvTuneKey) -> Self {
        let alignments = MatmulOperands {
            lhs: ElemAlignment::dense_rows(key.lhs_width),
            rhs: ElemAlignment::dense_rows(key.rhs_width),
            out: ElemAlignment::dense_rows(1),
        };
        Self {
            operation: key.operation,
            batch: key.batch,
            channels_in: key.channels_in,
            channels_out: key.channels_out,
            spatial: key.spatial.clone(),
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
