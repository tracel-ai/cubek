use crate::matmul::MatmulPartitioning;
use crate::matmul::staged_walk::MatmulStage;
use crate::matmul::{MatmulOperand, MatmulOperands, MatmulOrder, MatmulWorkItem};
use crate::staging::StageRing;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate};

/// How a kernel's cubes stage their boxes through `K`, as the component that places the stage's
/// reads states it: a staged cube for its ring, a routine for its own planes' or units' steps.
///
/// The value that component builds in its `estimate`; [`count`](Self::count) sets the estimate's
/// staging from it, written once for every one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulStagingCount {
    /// Elements of `K` one stage covers.
    pub depth: usize,
    /// Stages a cube holds at once.
    pub stages_in_flight: u32,
    /// Elements of each operand a plane touches contiguously over the cube's walk: a line a later
    /// stage finishes reading is one the cache still holds.
    pub runs: MatmulOperands<usize>,
    /// Times one cube reads each element of its box of each operand: once where it stages the box
    /// for all its units, once per unit that reads it where each reads its own; for the output,
    /// the passes a cube makes over it.
    pub readers: MatmulOperands<u64>,
    /// Atomic adds into shared memory one cube issues as it drains.
    pub drain_atomics: u64,
}

impl MatmulStagingCount {
    /// What a cube staging the box `partitioning` hands it through `ring` a step of `stage` at a
    /// time counts, bound in the order `orders` says: both operands read once a cube, in runs along
    /// their contiguous axis over the whole walk, and the output written once in runs of `out_run`
    /// elements.
    pub fn new(
        partitioning: &MatmulPartitioning,
        stage: &MatmulStage,
        orders: MatmulOperands<MatmulOrder>,
        ring: &StageRing,
        out_run: usize,
    ) -> Self {
        Self {
            depth: stage.extent(K),
            stages_in_flight: ring.stages_in_flight.get() as u32,
            runs: MatmulOperands {
                lhs: stage.walk_run(partitioning, MatmulOperand::Lhs, orders),
                rhs: stage.walk_run(partitioning, MatmulOperand::Rhs, orders),
                out: out_run,
            },
            readers: MatmulOperands {
                lhs: 1,
                rhs: 1,
                out: 1,
            },
            drain_atomics: 0,
        }
    }

    /// Set `estimate`'s staging to what one cube staging the box `partitioning` hands it this way
    /// moves and computes for `item`: it reads its box's rows of the lhs and columns of the rhs
    /// over the `K` its stages cover, as many times over as each has readers, and passes over its
    /// box of the output as many times as the output has readers. A box the problem's edge cuts
    /// short is still read, computed and written whole, since that is what its cube spends.
    pub fn count(
        &self,
        item: &MatmulWorkItem,
        partitioning: &MatmulPartitioning,
        estimate: &mut KernelEstimate,
    ) {
        let cube = &partitioning.cube_box();
        let (rows, columns) = (cube.extent(M) as u64, cube.extent(N) as u64);
        let stages = cube.extent(K).div_ceil(self.depth.max(1)) as u64;
        let depth = stages * self.depth as u64;
        let stream = |operand: MatmulOperand, elements: u64| {
            let size = item.elements.get(operand).size();
            GlobalStream {
                bytes: self.readers.get(operand) * elements * size as u64,
                run_bytes: self.runs.get(operand) * size,
            }
        };
        let reads = vec![
            stream(MatmulOperand::Lhs, rows * depth),
            stream(MatmulOperand::Rhs, depth * columns),
        ];
        let writes = vec![stream(MatmulOperand::Out, rows * columns)];
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * depth) as f64,
            reads,
            writes,
            stages,
            stages_in_flight: self.stages_in_flight,
            drain_atomics: self.drain_atomics,
        });
    }
}
