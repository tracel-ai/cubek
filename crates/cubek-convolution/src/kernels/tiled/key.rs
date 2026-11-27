use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{B, ConvBias, ConvProblem, ConvTensors, SpatialDim};
use cubecl::ir::ElemType;
use cubecl::tile::layout::ElemAlignment;
use cubecl::{self as cubecl, AutotuneKeyable};
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use serde::{Deserialize, Serialize};

/// What makes two convolutions want the same setting.
///
/// Every extent is exact, since a model has few layer shapes and anchoring would merge layers that
/// want different settings, except the batch: a caller varies it freely, so it is anchored to the
/// power of two above it. The widths are how wide each operand's alignment, and its view's start,
/// lets it be read, the output's excepted: the cube staged routine writes it one element at a time.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKeyable)]
pub struct CubeStagedConvTuneKey {
    pub operation: ConvolutionOperation,
    pub lhs: ElemType,
    pub rhs: ElemType,
    pub out: ElemType,
    pub sum: ElemType,
    pub batch: usize,
    pub groups: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub spatial: Vec<SpatialDim>,
    pub bias: ConvBias,
    pub lhs_width: usize,
    pub rhs_width: usize,
}

impl CubeStagedConvTuneKey {
    /// The key of the convolution `tensors` state on `hw`.
    pub fn of(tensors: &ConvTensors, hw: &Hardware) -> Self {
        Self::of_problem(&tensors.problem(), hw)
    }
}

impl CubeStagedConvTuneKey {
    /// The widest width a key tells apart is two to this power: a wider alignment shares the key.
    const MOST_WIDTH_POWER: u32 = 5;

    /// The key of `problem` on `hw`.
    pub(crate) fn of_problem(problem: &ConvProblem, hw: &Hardware) -> Self {
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
            groups: problem.groups,
            channels_in: problem.channels_in,
            channels_out: problem.channels_out,
            spatial: problem.spatial.clone(),
            bias: problem.bias,
            lhs_width: width(MatmulOperand::Lhs),
            rhs_width: width(MatmulOperand::Rhs),
        }
    }

    /// The batch `problem` is keyed by: the power of two above it, unless a launch of that many
    /// batches would put more cubes along the grid's third dimension (the batches beside the
    /// positions across) than the device spawns, which settings may then disagree on within one
    /// bucket.
    fn batch(problem: &ConvProblem, hw: &Hardware) -> usize {
        let free = problem.roles().free;
        let bucket = problem.batch.next_power_of_two();
        let grid = || free.outer.iter().chain([&free.across]);
        let rides_grid = grid().any(|&dimension| dimension == B);
        let cubes = grid()
            .map(|&dimension| match dimension == B {
                true => bucket,
                false => problem.extent(dimension),
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
            groups: key.groups,
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
            bias: key.bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::benchmarks::Conv2dProblem;
    use crate::kernels::tiled::routines::test_support::{dense, nvidia, problems};
    use crate::kernels::tiled::{
        CubeStagedConvCandidate, CubeStagedConvPrepared, CubeStagedConvRoutine,
    };
    use cubek_std::devices::topology;
    use cubek_std::routine::Routine;

    fn key(problem: Conv2dProblem) -> CubeStagedConvTuneKey {
        CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&problem), &nvidia())
    }

    /// A view starting at an odd element narrows the key's width, and the launch the key's
    /// problem prepares is the one the real problem prepares, so the candidate tuned is the one
    /// run.
    #[test]
    fn a_key_holds_the_width_a_view_start_leaves() {
        let hw = nvidia();
        let aligned = ConvProblem::from(&dense());
        let mut shifted = aligned.clone();
        shifted.start_bytes.lhs = 2;
        let (aligned_key, shifted_key) = (
            CubeStagedConvTuneKey::of_problem(&aligned, &hw),
            CubeStagedConvTuneKey::of_problem(&shifted, &hw),
        );
        assert_ne!(aligned_key, shifted_key);
        for (problem, key) in [(&aligned, &aligned_key), (&shifted, &shifted_key)] {
            for candidate in CubeStagedConvCandidate::list(&hw) {
                let real = CubeStagedConvRoutine::prepare(&hw, problem, candidate.settings())
                    .ok()
                    .map(|(blueprint, grid)| {
                        CubeStagedConvPrepared::new(problem, blueprint, grid.launch.cube_dim)
                    });
                assert_eq!(real, candidate.prepared(key), "{}", candidate.settings());
            }
        }
    }

    /// The problem a key describes is keyed as the key is.
    #[test]
    fn a_key_describes_a_problem_keyed_as_itself() {
        let hw = nvidia();
        for (name, problem) in problems() {
            let key = CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&problem), &hw);
            assert_eq!(
                CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&key), &hw),
                key,
                "{name}"
            );
        }
    }

    /// A grouped convolution keys apart from the dense one of its channels, and its key describes
    /// a problem keyed as the key is.
    #[test]
    fn a_grouped_key_describes_a_problem_keyed_as_itself() {
        let hw = nvidia();
        let dense = ConvProblem::from(&dense());
        let grouped = ConvProblem {
            groups: 2,
            ..dense.clone()
        };
        let key = CubeStagedConvTuneKey::of_problem(&grouped, &hw);
        assert_ne!(key, CubeStagedConvTuneKey::of_problem(&dense, &hw));
        assert_eq!(
            CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&key), &hw),
            key
        );
    }

    /// Batches that fill the grid's third dimension with the positions across are keyed exactly,
    /// since settings may disagree within a bucket there; below that they share their power of
    /// two.
    #[test]
    fn a_batch_is_keyed_exactly_only_where_it_fills_the_grid() {
        let batched = |batch| {
            key(Conv2dProblem {
                input_shape: [batch, 64, 58, 58],
                ..dense()
            })
        };
        assert_eq!(batched(5), batched(7));
        assert_ne!(batched(1200), batched(1201));
    }

    /// The grid limit is the device's: a device spawning few cubes along the third dimension
    /// keys a small batch exactly.
    #[test]
    fn a_batch_is_keyed_by_the_device_grid_limit() {
        let mut hw = nvidia();
        topology(&mut hw).max_cube_count.2 = 64;
        let batched = |batch| {
            let problem = Conv2dProblem {
                input_shape: [batch, 64, 58, 58],
                ..dense()
            };
            CubeStagedConvTuneKey::of_problem(&ConvProblem::from(&problem), &hw)
        };
        assert_ne!(batched(5), batched(7));
    }

    /// A weight backward's batch is contracted, never a grid dimension, so it is bucketed however
    /// large.
    #[test]
    fn a_contracted_batch_is_keyed_by_its_bucket() {
        let batched = |batch| {
            key(Conv2dProblem {
                operation: ConvolutionOperation::BackwardWeight,
                input_shape: [batch, 64, 58, 58],
                ..dense()
            })
        };
        assert_eq!(batched(1200), batched(1201));
    }
}
