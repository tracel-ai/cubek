//! What one launch of the backward pass moves and computes, counted from the boxes its owners
//! cover ([`WalkCount`]), and what the whole problem asks for.

use crate::kernels::AttentionBackwardProblem;
use cubecl::ir::ElemType;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Dimension, Partitioning, Space};
use cubecl::tune::Work;
use cubek_microkernel::attention::{D, DV, G, Q, S};
use cubek_microkernel::attention_backward::AttentionBackwardOperand;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, StageRing};
use cubek_std::routine::EstimatedProblem;

/// The problem's arithmetic is its five products on every pair of a query and a key — the score
/// and the gradient against the probabilities, then the values', the keys' and the query's
/// gradients — and its bytes every operand's, each read or written once.
impl EstimatedProblem for AttentionBackwardProblem {
    fn demand(&self) -> Work {
        let pairs = self.batch * self.num_kv_heads * self.group * self.seq_q * self.seq_kv;
        Work {
            compute_ops: 2 * pairs * (3 * self.head_dim + 2 * self.val_dim),
            bytes: AttentionBackwardOperand::ALL
                .iter()
                .map(|&operand| self.bytes_of(operand, self.elements().get(operand)))
                .sum(),
        }
    }
}

impl AttentionBackwardProblem {
    /// Bytes `operand` holds over the whole problem, stored at `element`.
    fn bytes_of(&self, operand: AttentionBackwardOperand, element: ElemType) -> usize {
        let space = self.space();
        let elements: usize = operand
            .dimensions()
            .iter()
            .map(|&dimension| space.extent(dimension))
            .product();
        elements * element.size()
    }
}

/// One operand a launch moves through global memory, at the element it moves it in: the stored
/// one, or the `f32` of a sum the launch adds into or casts out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WalkStream {
    pub operand: AttentionBackwardOperand,
    pub element: ElemType,
    /// Times one owner moves each element of its box of the operand: once, or once for each of
    /// its planes that reads it where it lies.
    pub movers: u64,
}

impl WalkStream {
    /// `operand` moved once by its owner, at `element`.
    pub fn once(operand: AttentionBackwardOperand, element: ElemType) -> Self {
        Self {
            operand,
            element,
            movers: 1,
        }
    }
}

/// What one owner stages in shared memory as it walks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WalkSharing {
    /// The walked operands it stages, each stage written once and read by `readers` planes.
    pub walked: Vec<AttentionBackwardOperand>,
    pub readers: u64,
    /// The owned operands it stages once and reads back every stage.
    pub owned: [AttentionBackwardOperand; 2],
    /// Bytes its planes write into their windows once a stage, and read back.
    pub window_bytes: u64,
    /// Atomic adds into shared memory it issues once a stage, its planes meeting.
    pub drain_atomics: u64,
}

impl WalkSharing {
    /// A walk owned by the queries: each stage of keys and values read by `readers` planes, the
    /// owner's queries and output's gradient staged once, `window_bytes` through its planes'
    /// windows a stage.
    pub fn query_walk(readers: usize, window_bytes: usize) -> Self {
        use AttentionBackwardOperand as Operand;
        Self {
            walked: vec![Operand::Key, Operand::Value],
            readers: readers as u64,
            owned: [Operand::Query, Operand::OutGradient],
            window_bytes: window_bytes as u64,
            drain_atomics: 0,
        }
    }

    /// A walk owned by the keys: each stage of queries and output's gradient read by `readers`
    /// planes, the owner's keys and values staged once, `window_bytes` through its planes'
    /// windows a stage.
    pub fn key_walk(readers: usize, window_bytes: usize) -> Self {
        use AttentionBackwardOperand as Operand;
        Self {
            walked: vec![Operand::Query, Operand::OutGradient],
            readers: readers as u64,
            owned: [Operand::Key, Operand::Value],
            window_bytes: window_bytes as u64,
            drain_atomics: 0,
        }
    }
}

/// What one launch of the backward pass moves and computes, an owner at a time: each owner — a
/// unit, a plane or a cube, as the launch's partitioning distributes them — covers a box of the
/// problem, walks it a step at a time and moves its box of each operand. The cube's share is the
/// owners' total over the launch's cubes, so cubes holding several owners, and cubes holding
/// none, are both counted.
///
/// A box the problem's edge or a causal mask cuts short is counted whole, as the forward's
/// estimates count theirs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WalkCount {
    /// The box one owner covers: its rows and everything it walks.
    pub owner: Space,
    /// Owners the launch runs.
    pub owners: u64,
    /// Each dimension the owner walks and the extent one step of it covers.
    pub steps: Vec<(Dimension, usize)>,
    /// Steps an owner holds at once.
    pub stages_in_flight: u32,
    /// Arithmetic operations one owner performs.
    pub operations: u64,
    pub reads: Vec<WalkStream>,
    pub writes: Vec<WalkStream>,
    /// What it stages in shared memory; `None` for owners that read where the operands lie.
    pub sharing: Option<WalkSharing>,
}

impl WalkCount {
    /// The box one of `partitioning`'s instances of `scope` covers, and how many of them the
    /// launch runs.
    ///
    /// # Panics
    ///
    /// Where no level of `partitioning` distributes over `scope`.
    pub fn owners(partitioning: &Partitioning, scope: ComputeScope) -> (Space, u64) {
        let owner = partitioning.box_of(scope);
        let space = partitioning.space();
        let owners = space
            .dimensions()
            .map(|dimension| {
                space
                    .extent(dimension)
                    .div_ceil(owner.extent(dimension).max(1))
            })
            .product::<usize>();
        (owner, owners as u64)
    }

    /// A walk owned by the queries: each owner walks every key a step of `seq_kv` at a time,
    /// reads its queries, their output, its gradient and their logsumexp, and every key and
    /// value, and writes their gradient and dot products, on three products: the score, the
    /// gradient against the probabilities and the query's gradient.
    pub fn query_walk(
        problem: &AttentionBackwardProblem,
        partitioning: &Partitioning,
        scope: ComputeScope,
        seq_kv: usize,
    ) -> Self {
        use AttentionBackwardOperand as Operand;
        let (owner, owners) = Self::owners(partitioning, scope);
        let elements = problem.elements();
        let once = |operand| WalkStream::once(operand, elements.get(operand));
        Self {
            operations: products(&owner, &[D, DV, D]),
            owner,
            owners,
            steps: vec![(S, seq_kv)],
            stages_in_flight: 1,
            reads: [
                Operand::Query,
                Operand::Key,
                Operand::Value,
                Operand::Out,
                Operand::OutGradient,
                Operand::Logsumexp,
            ]
            .map(once)
            .into(),
            writes: vec![once(Operand::QueryGradient), once(Operand::Delta)],
            sharing: None,
        }
    }

    /// A walk owned by the keys: each owner walks every query of the heads it holds, a step of
    /// `seq_q` at a time and a head after another, reads its keys and values, and the queries,
    /// the output's gradient, the logsumexp and the dot products of each, and writes its keys'
    /// and values' gradients — at the sum's element where it adds into sums, the heads of a group
    /// apart — on four products: the score, the gradient against the probabilities and the
    /// values' and the keys' gradients.
    pub fn key_walk(
        problem: &AttentionBackwardProblem,
        partitioning: &Partitioning,
        scope: ComputeScope,
        seq_q: usize,
        written: KeyGradientElement,
    ) -> Self {
        use AttentionBackwardOperand as Operand;
        let (owner, owners) = Self::owners(partitioning, scope);
        let elements = problem.elements();
        let once = |operand| WalkStream::once(operand, elements.get(operand));
        let gradient = match written {
            KeyGradientElement::Stored => problem.element,
            KeyGradientElement::Summed => AttentionBackwardProblem::SUM,
        };
        Self {
            operations: products(&owner, &[D, DV, DV, D]),
            owner,
            owners,
            steps: vec![(Q, seq_q), (G, 1)],
            stages_in_flight: 1,
            reads: [
                Operand::Query,
                Operand::Key,
                Operand::Value,
                Operand::OutGradient,
                Operand::Logsumexp,
                Operand::Delta,
            ]
            .map(once)
            .into(),
            writes: [Operand::KeyGradient, Operand::ValueGradient]
                .map(|operand| WalkStream::once(operand, gradient))
                .into(),
            sharing: None,
        }
    }

    /// This walk owned by the keys also adding its share of the queries' gradient into their
    /// `f32` sum: one product more on every pair, and its box of the queries' gradient written at
    /// the sum's element, which every owner of keys those queries see adds into.
    pub fn adding_query_gradient(mut self) -> Self {
        self.operations += products(&self.owner, &[D]);
        self.writes.push(WalkStream::once(
            AttentionBackwardOperand::QueryGradient,
            AttentionBackwardProblem::SUM,
        ));
        self
    }

    /// A launch each of whose planes writes its queries' dot products of the output with its
    /// gradient: it reads both, a multiply and an add an element of the value dim.
    pub fn delta(problem: &AttentionBackwardProblem, partitioning: &Partitioning) -> Self {
        use AttentionBackwardOperand as Operand;
        let (owner, owners) = Self::owners(partitioning, ComputeScope::Plane);
        let elements = problem.elements();
        let once = |operand| WalkStream::once(operand, elements.get(operand));
        Self {
            operations: 2 * box_elements(&owner, Operand::Out),
            owner,
            owners,
            steps: Vec::new(),
            stages_in_flight: 1,
            reads: vec![once(Operand::Out), once(Operand::OutGradient)],
            writes: vec![once(Operand::Delta)],
            sharing: None,
        }
    }

    /// A launch each of whose owners casts its box of `operands`, out of their sums: one read of
    /// the sum's element and one write of the stored element an element, one operation each.
    pub fn cast(
        problem: &AttentionBackwardProblem,
        (owner, owners): (Space, u64),
        operands: &[AttentionBackwardOperand],
    ) -> Self {
        let stream = |element| {
            operands
                .iter()
                .map(|&operand| WalkStream::once(operand, element))
                .collect()
        };
        Self {
            operations: operands
                .iter()
                .map(|&operand| box_elements(&owner, operand))
                .sum(),
            owner,
            owners,
            steps: Vec::new(),
            stages_in_flight: 1,
            reads: stream(AttentionBackwardProblem::SUM),
            writes: stream(problem.element),
            sharing: None,
        }
    }

    /// This walk with its owners staging as `sharing` says, holding the stages of `ring`.
    pub fn staged(self, ring: &StageRing, sharing: WalkSharing) -> Self {
        Self {
            stages_in_flight: ring.stages_in_flight.get() as u32,
            sharing: Some(sharing),
            ..self
        }
    }

    /// The steps one owner walks.
    fn stages(&self) -> u64 {
        self.steps
            .iter()
            .map(|&(dimension, step)| self.owner.extent(dimension).div_ceil(step.max(1)) as u64)
            .product()
    }

    /// Set `estimate`'s staging to one cube's share of what this launch's owners move and
    /// compute, and its demand to the launch's own: every operand it touches moved once, and
    /// every owner's arithmetic. The estimate's cubes are the launch's; where they are known only
    /// at launch, every owner is its own cube.
    pub fn count(&self, problem: &AttentionBackwardProblem, estimate: &mut KernelEstimate) {
        let cubes = estimate.cubes.unwrap_or(self.owners).max(1);
        let share = |per_owner: u64| (per_owner * self.owners).div_ceil(cubes);
        let walked = |dimension| self.steps.iter().find(|&&(walked, _)| walked == dimension);
        let stream = |stream: &WalkStream| {
            let size = stream.element.size();
            let innermost = stream.operand.innermost();
            let run = match walked(innermost) {
                Some(&(_, step)) => step,
                None => self.owner.extent(innermost),
            };
            GlobalStream {
                bytes: share(
                    stream.movers * box_elements(&self.owner, stream.operand) * size as u64,
                ),
                run_bytes: run * size,
            }
        };
        let reads: Vec<GlobalStream> = self.reads.iter().map(stream).collect();
        let stages = self.stages();
        let (shared_reads, shared_writes, drain_atomics) = match &self.sharing {
            Some(sharing) => {
                let staged = |operands: &[AttentionBackwardOperand]| -> u64 {
                    self.reads
                        .iter()
                        .zip(&reads)
                        .filter(|(read, _)| operands.contains(&read.operand))
                        .map(|(_, stream)| stream.bytes)
                        .sum()
                };
                let (walked, owned) = (staged(&sharing.walked), staged(&sharing.owned));
                let windows = share(stages * sharing.window_bytes);
                (
                    sharing.readers * walked + stages * owned + windows,
                    walked + owned + windows,
                    share(stages * sharing.drain_atomics),
                )
            }
            None => (0, 0, 0),
        };
        let touched = |streams: &[WalkStream]| -> usize {
            streams
                .iter()
                .map(|stream| problem.bytes_of(stream.operand, stream.element))
                .sum()
        };
        estimate.demand = Some(Work {
            compute_ops: (self.operations * self.owners) as usize,
            bytes: touched(&self.reads) + touched(&self.writes),
        });
        estimate.staging = Some(CubeStaging {
            operations: share(self.operations) as f64,
            writes: self.writes.iter().map(stream).collect(),
            reads,
            stages,
            stages_in_flight: self.stages_in_flight,
            drain_atomics,
            shared_reads,
            shared_writes,
        });
    }
}

/// The element a walk owned by the keys writes their gradients in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyGradientElement {
    /// The stored element: the walk writes the gradients.
    Stored,
    /// The sum's: the walk adds into sums a cast reads.
    Summed,
}

/// Elements of `operand` within `owner`'s box.
fn box_elements(owner: &Space, operand: AttentionBackwardOperand) -> u64 {
    operand
        .dimensions()
        .iter()
        .map(|&dimension| owner.extent(dimension) as u64)
        .product()
}

/// Operations of the products one owner of `owner`'s box runs on every pair of its queries and
/// keys, each contracting the dimension `contracted` names: a multiply and an add per element of
/// it.
fn products(owner: &Space, contracted: &[Dimension]) -> u64 {
    let pairs = (owner.extent(Q) * owner.extent(G) * owner.extent(S)) as u64;
    let depth: u64 = contracted
        .iter()
        .map(|&dimension| owner.extent(dimension) as u64)
        .sum();
    2 * pairs * depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{
        AttentionBackwardCandidate, AttentionBackwardKey, AttentionBackwardLaunch,
        CastKeyGradientRoutine, CubeStagedAttentionBackward, CubeStagedKeyGradientSumRoutine,
        CubeStagedKeyWalkSettings, HypercubeAttentionBackward, PlaneFragmentAttentionBackward,
        UnitAttentionBackward, UnitQueryGradientRoutine, UnitQueryGradientRoutineSettings,
    };
    use core::num::NonZeroUsize;
    use cubecl::features::MmaConfig;
    use cubecl::ir::FloatKind;
    use cubek_microkernel::attention::AttentionVisibility;
    use cubek_microkernel::attention_backward::key_gradient::cube::CubeKeyGradientMicrokernelSettings;
    use cubek_microkernel::attention_backward::query_gradient::unit::UnitQueryGradientMicrokernelSettings;
    use cubek_std::{Hardware, devices};

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// An H100 offering the fragment API's three `f16` shapes and the manual `16 × 8 × 16`, summed
    /// in `f32`: every setting's instruction.
    fn h100() -> Hardware {
        let shape = |(m, n, k)| MmaConfig {
            a_type: F16,
            b_type: F16,
            cd_type: F32,
            m,
            n,
            k,
        };
        let hw = devices::offering_cmma(
            devices::h100(),
            [(16, 16, 16), (32, 8, 16), (8, 32, 16)].map(shape),
        );
        devices::offering_mma(hw, [shape((16, 8, 16))])
    }

    /// An `f16` key of 8 query heads over 2 key heads of dim 64, 300 queries over 300 keys, in
    /// their buckets.
    fn key() -> AttentionBackwardKey {
        AttentionBackwardKey::new(
            F16,
            8,
            2,
            64,
            64,
            300,
            300,
            1,
            AttentionVisibility::Attended,
        )
    }

    fn problem() -> AttentionBackwardProblem {
        AttentionBackwardProblem::try_from(&key()).unwrap()
    }

    /// Every setting of `strategy` that prepares on `hw` states, for each launch, its staging,
    /// its work and its own demand; the settings that prepare launch as many kernels as each of
    /// `launches`, and no other count.
    fn assert_every_launch_estimated<S: AttentionBackwardLaunch + Copy>(
        strategy: S,
        hw: &Hardware,
        launches: &[usize],
    ) {
        let estimated: Vec<_> = AttentionBackwardCandidate::list(strategy, hw)
            .iter()
            .filter_map(|candidate| {
                let estimates = candidate.estimates(&key())?;
                Some((candidate.settings().to_string(), estimates))
            })
            .collect();
        let mut counts: Vec<usize> = estimated
            .iter()
            .map(|(_, estimates)| estimates.len())
            .collect();
        counts.sort_unstable();
        counts.dedup();
        assert_eq!(
            counts, launches,
            "the launch counts of the settings that prepare"
        );
        for (settings, estimates) in estimated {
            for estimate in estimates {
                assert!(estimate.staging.is_some(), "{settings}: no staging");
                assert!(estimate.work.is_some(), "{settings}: no work");
                assert!(estimate.demand.is_some(), "{settings}: no demand");
            }
        }
    }

    #[test]
    fn every_launch_of_the_unit_strategy_is_estimated() {
        assert_every_launch_estimated(UnitAttentionBackward, &devices::h100(), &[2]);
    }

    #[test]
    fn every_launch_of_the_plane_strategy_is_estimated() {
        assert_every_launch_estimated(PlaneFragmentAttentionBackward, &h100(), &[2]);
    }

    /// Two launches with the key walk's heads together, three with them apart.
    #[test]
    fn every_launch_of_the_cube_staged_strategy_is_estimated() {
        assert_every_launch_estimated(CubeStagedAttentionBackward, &h100(), &[2, 3]);
    }

    /// Three launches with the key walk's heads together, four with them apart.
    #[test]
    fn every_launch_of_the_hypercube_strategy_is_estimated() {
        assert_every_launch_estimated(HypercubeAttentionBackward, &h100(), &[3, 4]);
    }

    /// Bytes of operand `read` the launch `estimate` describes reads, over all its cubes.
    fn launch_reads(estimate: &KernelEstimate, read: usize) -> u64 {
        estimate.staging.as_ref().unwrap().reads[read].bytes * estimate.cubes.unwrap()
    }

    /// Every unit of the unit walk owned by the queries reads every key where it lies, so the
    /// launch reads them once for each unit's queries, not once a cube: two queries a unit read
    /// the keys half as many times as one.
    #[test]
    fn a_unit_walk_reads_the_keys_once_a_unit() {
        let (hw, problem) = (devices::h100(), problem());
        let key_reads = |queries| {
            let settings = UnitQueryGradientRoutineSettings {
                microkernel: UnitQueryGradientMicrokernelSettings::new(queries, 8),
                planes: NonZeroUsize::new(2).unwrap(),
            };
            let walk = UnitQueryGradientRoutine
                .prepare(&hw, &problem, &settings)
                .unwrap();
            launch_reads(&UnitQueryGradientRoutine.estimate(&problem, &walk), 1)
        };
        // A key head's keys, read by each of its group's heads' queries.
        let keys = (problem.seq_kv * problem.head_dim * 2) as u64;
        let query_rows = (problem.num_kv_heads * problem.group * problem.seq_q) as u64;
        assert_eq!(key_reads(1), query_rows * keys, "a query a unit");
        assert_eq!(key_reads(2), query_rows / 2 * keys, "two queries a unit");
    }

    /// Only the cubes on a group's first head cast the keys' sums, so the cast reads each sum
    /// once, however many heads the summing walk set apart.
    #[test]
    fn the_cast_of_the_key_sums_reads_each_sum_once() {
        let (hw, problem) = (h100(), problem());
        let settings = CubeStagedKeyWalkSettings {
            microkernel: CubeKeyGradientMicrokernelSettings::DEFAULT,
            ring: cubek_std::microkernel::StageRing::SINGLE,
        };
        let sums = CubeStagedKeyGradientSumRoutine
            .prepare(&hw, &problem, &settings)
            .unwrap();
        let cast = CastKeyGradientRoutine.prepare(&sums);
        let estimate = CastKeyGradientRoutine.estimate(&problem, &cast);
        let row = |width: usize| (problem.num_kv_heads * problem.seq_kv * width * 4) as u64;
        assert_eq!(
            launch_reads(&estimate, 0),
            row(problem.head_dim),
            "the keys' sum"
        );
        assert_eq!(
            launch_reads(&estimate, 1),
            row(problem.val_dim),
            "the values' sum"
        );
    }
}
