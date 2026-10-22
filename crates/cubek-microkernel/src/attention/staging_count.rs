use crate::attention::{
    AttentionOperand, AttentionOperands, AttentionPartitioning, AttentionWorkItem, G, Q, S,
};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate};

/// How the cubes of a kernel of this kind walk the keys, as the component that places the
/// stage's reads states it: the unit routine for its units' steps, the plane routine for its
/// planes' own stages, the cube microkernel for its ring.
///
/// The value that component builds in its `estimate`; [`count`](Self::count) sets the estimate's
/// staging from it, written once for every one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionStagingCount {
    /// Keys one stage covers.
    pub seq_kv: usize,
    /// Stages a reader holds at once.
    pub stages_in_flight: u32,
    /// Elements of each operand a plane touches contiguously over the walk.
    pub runs: AttentionOperands<usize>,
    /// Times one cube reads each element of its box of each operand: once where it stages the box
    /// for all its units, once per plane or unit that reads its own; for the output, once.
    pub readers: AttentionOperands<u64>,
}

impl AttentionStagingCount {
    /// Set `estimate`'s staging to what one cube walking its box of `partitioning` this way moves
    /// and computes for `item`: its rows' queries and outputs, and every key and value the buffers
    /// hold, each as many times over as it has readers. The attended keys are the device's, so a
    /// cube is counted against the capacity; a box the problem's edge cuts short is counted whole.
    pub fn count(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        estimate: &mut KernelEstimate,
    ) {
        let cube = partitioning.cube_box();
        let rows = (cube.extent(Q) * cube.extent(G)) as u64;
        let keys = cube.extent(S) as u64;
        let (head, value) = (item.head_dim as u64, item.val_dim as u64);
        let stream = |operand: AttentionOperand, elements: u64| {
            let size = item.elements.get(operand).size();
            GlobalStream {
                bytes: self.readers.get(operand) * elements * size as u64,
                run_bytes: self.runs.get(operand) * size,
            }
        };
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * keys * (head + value)) as f64,
            reads: vec![
                stream(AttentionOperand::Query, rows * head),
                stream(AttentionOperand::Key, keys * head),
                stream(AttentionOperand::Value, keys * value),
            ],
            writes: vec![stream(AttentionOperand::Out, rows * value)],
            stages: keys.div_ceil(self.seq_kv.max(1) as u64),
            stages_in_flight: self.stages_in_flight,
            drain_atomics: 0,
        });
    }
}
