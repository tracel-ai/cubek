//! The problem a routine computes, and the axis its queries are read along.

use crate::kernels::{AttentionVisibility, PackedLayout};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{
    AttentionOperand, AttentionOperands, AttentionReading, AttentionWorkItem, B, D, DV, G, KV, Q, S,
};
use cubek_std::routine::EstimatedProblem;

/// `out[b, q, h] = softmax(query[b, h, q] · key[b, h / group]ᵀ · scale) · value[b, h / group]`,
/// as the host describes it, its queries along the axis [`QueryAxis`] says: a single position's
/// `group` heads are `group` queries of one head.
///
/// The queries are the last `queries` positions of the attended keys, whose count `len` only the
/// device holds: `keys` is the capacity the key and value buffers are allocated to, of which the
/// kernel reads the first `len`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionRoutineProblem {
    pub batch: usize,
    pub num_kv_heads: usize,
    /// Query heads per key head.
    pub group: usize,
    pub seq_q: usize,
    /// The keys the buffers hold, attended or not.
    pub seq_kv: usize,
    pub head_dim: usize,
    pub val_dim: usize,
    /// Which of the attended keys a query sees.
    pub visibility: AttentionVisibility,
    /// The element each operand is stored in.
    pub elements: AttentionOperands<ElemType>,
    /// The alignment of each operand's buffer in elements, over the axes this kind names.
    pub elem_alignments: AttentionOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: AttentionOperands<u64>,
    /// How the keys and values lie.
    pub cache: AttentionCacheLayout,
}

/// How an attention's keys and values lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionCacheLayout {
    /// As sequences, `[b, kv, s, ·]`.
    Sequences,
    /// In two pools of pages of `page_size` positions, each sequence's pages named in order by a
    /// table.
    Pages { page_size: usize },
    /// Packed into words of codebook indices, as the layout states.
    Packed(PackedLayout),
}

impl AttentionRoutineProblem {
    /// The element every microkernel of this kind sums in, whatever it reads and writes.
    pub const SUM: ElemType = ElemType::Float(FloatKind::F32);

    /// The widest capacity [`capacity_classes`](Self::capacity_classes) lists: a multiple of
    /// every step of keys a routine walks.
    pub const WIDEST_CAPACITY_CLASS: usize = 2048;

    /// A multiple of every step of keys a routine walks, at every setting it lists: a cache
    /// holding a multiple of it binds its keys as whole steps, so no launch over it masks its
    /// reads. A capacity off it compiles the bounded launch, which masks every read of its walk
    /// rather than its last step's alone.
    pub const WIDEST_STEP: usize = 256;

    /// The capacities, up to `limit`, a caller compiling ahead of time runs a prefill at: every
    /// power of two from 2 to [`WIDEST_CAPACITY_CLASS`](Self::WIDEST_CAPACITY_CLASS), the last
    /// one inside `limit` standing for it.
    ///
    /// A routine's launch compiles apart where the capacity is not a multiple of its step of
    /// keys, the walk's last step then bounded. Two is a multiple of no step wider than it, and
    /// the widest class of every step, so a walk over the classes compiles both launches of every
    /// setting; the powers between land the tuning key's buckets ([`AttentionKey`](crate::kernels::tune::key::AttentionKey)).
    pub fn capacity_classes(limit: usize) -> Vec<usize> {
        let mut classes = Vec::new();
        for class in std::iter::successors(Some(2usize), |class| Some(class * 2)) {
            classes.push(class.min(limit));
            if class >= limit || class >= Self::WIDEST_CAPACITY_CLASS {
                break;
            }
        }
        classes
    }

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> AttentionWorkItem {
        AttentionWorkItem {
            elements: self.elements,
            sum: Self::SUM,
            reading: match self.cache {
                AttentionCacheLayout::Pages { .. } => AttentionReading::AsStored,
                AttentionCacheLayout::Sequences | AttentionCacheLayout::Packed(_) => {
                    AttentionReading::Rounded
                }
            },
            page_size: match self.cache {
                AttentionCacheLayout::Pages { page_size } => Some(page_size),
                AttentionCacheLayout::Sequences | AttentionCacheLayout::Packed(_) => None,
            },
            group: self.group,
            head_dim: self.head_dim,
            val_dim: self.val_dim,
        }
    }

    /// Query heads.
    pub fn heads(&self) -> usize {
        self.num_kv_heads * self.group
    }

    /// Every axis of the problem at its extent: what a routine partitions.
    pub fn space(&self) -> Space {
        Space::new(&[
            (B, self.batch),
            (KV, self.num_kv_heads),
            (G, self.group),
            (Q, self.seq_q),
            (S, self.seq_kv),
            (D, self.head_dim),
            (DV, self.val_dim),
        ])
    }

    /// A problem of dense operands of `element` throughout, their views starting where their
    /// buffers do: what a routine's pure tests prepare against.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense(
        (batch, heads, kv_heads): (usize, usize, usize),
        (queries, keys): (usize, usize),
        (head, value): (usize, usize),
        element: ElemType,
    ) -> Self {
        let aligned = |innermost: usize| ElemAlignment::new(&[1, innermost], &[innermost, 1]);
        Self {
            batch,
            num_kv_heads: kv_heads,
            group: heads / kv_heads,
            seq_q: queries,
            seq_kv: keys,
            head_dim: head,
            val_dim: value,
            visibility: AttentionVisibility::Attended,
            elements: AttentionOperands {
                query: element,
                key: element,
                value: element,
                out: element,
            },
            elem_alignments: AttentionOperands {
                query: aligned(head),
                key: aligned(head),
                value: aligned(value),
                // The output's heads are adjacent, so a query head's row starts a row apart.
                out: aligned(value),
            },
            start_bytes: AttentionOperands {
                query: 0,
                key: 0,
                value: 0,
                out: 0,
            },
            cache: AttentionCacheLayout::Sequences,
        }
    }

    /// Elements `operand` holds.
    pub fn elements_of(&self, operand: AttentionOperand) -> usize {
        let (queries, keys) = (
            self.batch * self.heads() * self.seq_q,
            self.batch * self.num_kv_heads * self.seq_kv,
        );
        match operand {
            AttentionOperand::Query => queries * self.head_dim,
            AttentionOperand::Key => keys * self.head_dim,
            AttentionOperand::Value => keys * self.val_dim,
            AttentionOperand::Out => queries * self.val_dim,
        }
    }
}

impl EstimatedProblem for AttentionRoutineProblem {
    /// Two operations per multiply-add of both contractions, every query against every key the
    /// buffers hold, and every operand's elements once. The attended keys are the device's: a
    /// problem is sized to its capacity, and one whose queries see fewer to all of it.
    fn demand(&self) -> Work {
        let pairs = self.batch * self.heads() * self.seq_q * self.seq_kv;
        Work {
            compute_ops: 2 * pairs * (self.head_dim + self.val_dim),
            bytes: AttentionOperand::ALL
                .iter()
                .map(|&operand| self.elements_of(operand) * self.elements.get(operand).size())
                .sum(),
        }
    }
}

/// What a routine that reads float keys and values declines on a packed cache.
pub const ATTENTION_PACKED_UNREAD: Refusal = Refusal::Declined {
    why: "this attention routine reads float keys and values, not a packed cache",
};

/// What a problem's query axis `Q` walks: the query positions of one head, or the heads of one key
/// head's group at a single position.
///
/// A single position of a group of heads is the same attention as each head at that position: it
/// sees every attended key, or the last of them a window spans, and the heads of a group read the
/// same keys. Walked along `Q`, the group fills the part of a fragment one position would leave
/// empty, and its keys are walked once for all of it; the heads are read at their own stride, so
/// nothing is copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryAxis {
    /// The query positions of one head, the heads of a group beside them.
    Positions,
    /// A single position's group of heads.
    GroupHeads,
}

impl QueryAxis {
    /// The query axis of an attention of `queries` positions.
    pub fn new(queries: usize) -> Self {
        match queries {
            1 => Self::GroupHeads,
            _ => Self::Positions,
        }
    }

    /// An attention of `group` heads a key head and `queries` positions seeing the keys
    /// `visibility` says, as a problem along this axis states it: the heads beside each query,
    /// the queries, and which keys a query sees.
    pub fn read(
        self,
        group: usize,
        queries: usize,
        visibility: AttentionVisibility,
    ) -> (usize, usize, AttentionVisibility) {
        match self {
            Self::Positions => (group, queries, visibility),
            Self::GroupHeads => (1, group * queries, visibility.of_last_position()),
        }
    }

    /// `tensor`'s heads split into a group per key head over the same buffer, the queries along
    /// this axis: `[b, kv, g, q, x]`, or where it walks the group's heads `[b, kv, q, g, x]`, the
    /// single position and the heads trading places. `dims` names which of the tensor's four dims
    /// are the batch, the heads, the queries and the last.
    pub(crate) fn view(self, tensor: &PlainTensor, group: usize, dims: [usize; 4]) -> PlainTensor {
        let (shape, strides) = (&tensor.binding.shape, &tensor.binding.strides);
        let [batch, heads, queries, last] = dims;
        let (heads_beside, along_queries) = match self {
            Self::Positions => ((group, strides[heads]), (shape[queries], strides[queries])),
            Self::GroupHeads => ((shape[queries], strides[queries]), (group, strides[heads])),
        };
        let view_shape = vec![
            shape[batch],
            shape[heads] / group,
            heads_beside.0,
            along_queries.0,
            shape[last],
        ];
        let view_strides = vec![
            strides[batch],
            strides[heads] * group,
            heads_beside.1,
            along_queries.1,
            strides[last],
        ];
        tensor.viewed(Metadata::new(view_shape, view_strides))
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::test_support::{offering, qwen};
    use crate::kernels::*;
    use cubek_microkernel::attention::plane::{
        PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
    };
    use cubek_microkernel::attention::{AttentionMicrokernel, S};
    use cubek_std::Hardware;
    use cubek_std::devices;

    /// Every power of two up to the widest class, the last inside the limit standing for it.
    #[test]
    fn capacity_classes_are_the_powers_of_two_to_the_widest() {
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(1 << 20),
            vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        );
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(100),
            vec![2, 4, 8, 16, 32, 64, 100]
        );
        assert_eq!(
            AttentionRoutineProblem::capacity_classes(64),
            vec![2, 4, 8, 16, 32, 64]
        );
    }

    /// The widest step is a multiple of every step of keys a routine walks, at every setting it
    /// lists on every device the fleet names and on an M2 Pro offering its square instruction, so
    /// a cache of whole widest steps is read unmasked; and the widest class is a multiple of it,
    /// so the classes compile every setting's launch over whole steps as well as its bounded one.
    #[test]
    fn every_step_of_keys_divides_the_widest_class() {
        assert!(
            AttentionRoutineProblem::WIDEST_CAPACITY_CLASS
                .is_multiple_of(AttentionRoutineProblem::WIDEST_STEP)
        );
        let fleet = [
            offering(devices::m2_pro(), 8),
            devices::cpu_12(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::m2_pro(),
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::h100(),
        ];
        let prepared = [
            fleet
                .iter()
                .map(|hw| assert_steps_divide(UnitAttentionRoutine, hw))
                .sum::<usize>(),
            fleet
                .iter()
                .map(|hw| assert_steps_divide(PlaneFragmentAttentionRoutine, hw))
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(
                        CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
                        hw,
                    )
                })
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(HypercubeLastArrivalShuffleAttentionRoutine::new(), hw)
                })
                .sum(),
            fleet
                .iter()
                .map(|hw| {
                    assert_steps_divide(HypercubeSecondLaunchShuffleAttentionRoutine::new(), hw)
                })
                .sum(),
        ];
        assert!(
            prepared.iter().all(|&count| count > 0),
            "prepared {prepared:?}"
        );
    }

    /// A problem read from pools of pages reads its operands as stored: a table binds no
    /// operand as words, so an `f32` cache is never read as tf32.
    #[test]
    fn a_paged_problem_reads_as_stored() {
        const F32: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32);
        let f32 = AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F32);
        let paged = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size: 16 },
            ..f32
        };
        assert_eq!(f32.work_item().readings().count(), 2);
        assert_eq!(paged.work_item().readings().collect::<Vec<_>>(), [F32]);
    }

    /// A cube whose planes read keys where they lie walks a paged cache a page at a time only
    /// where a page holds a whole number of its steps, more than one; a staging cube never does.
    #[test]
    fn a_cube_walks_pages_only_where_a_page_holds_several_whole_steps() {
        use cubek_std::routine::Routine;
        const F16: cubecl::ir::ElemType = cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16);
        let hw = devices::m2_pro();
        let dense = AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F16);
        let paged = |page_size| AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size },
            ..dense
        };
        let shuffle = &CubeShuffleAttentionRoutine.settings(&hw)[0];
        let page_steps = |problem: &AttentionRoutineProblem| {
            let (blueprint, _) = CubeShuffleAttentionRoutine
                .prepare(&hw, problem, shuffle)
                .expect("the decode step prepares");
            blueprint
                .microkernel
                .page_steps
                .map(core::num::NonZeroUsize::get)
        };
        let keys = CubeShuffleAttentionRoutine
            .prepare(&hw, &dense, shuffle)
            .expect("the decode step prepares")
            .0
            .microkernel
            .seq_kv();
        assert_eq!(page_steps(&dense), None);
        assert_eq!(page_steps(&paged(4 * keys)), Some(4));
        assert_eq!(page_steps(&paged(keys)), None);
        assert_eq!(page_steps(&paged(3 * keys / 2)), None);
        let staged = CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new();
        let staged_settings = &staged.settings(&hw)[0];
        let (blueprint, _) = staged
            .prepare(&hw, &paged(4 * keys), staged_settings)
            .expect("the decode step prepares");
        assert_eq!(blueprint.microkernel.page_steps, None);
    }

    /// Every routine but the two hypercube shuffles declines a packed cache, at every setting it
    /// lists; each hypercube shuffle prepares it.
    #[test]
    fn only_the_hypercube_shuffles_read_a_packed_cache() {
        fn reads<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
            routine: R,
            hw: &Hardware,
            problem: &AttentionRoutineProblem,
        ) -> bool {
            routine
                .settings(hw)
                .iter()
                .any(|settings| routine.prepare(hw, problem, settings).is_ok())
        }
        fn declines<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
            routine: R,
            hw: &Hardware,
            problem: &AttentionRoutineProblem,
        ) {
            for settings in routine.settings(hw) {
                assert_eq!(
                    routine.prepare(hw, problem, &settings).err(),
                    Some(ATTENTION_PACKED_UNREAD),
                    "{settings}"
                );
            }
        }
        let hw = offering(devices::h100(), 16);
        let layout = PackedLayout {
            bits: cubek_microkernel::attention::packed_cache::KvBitWidths { key: 4, value: 4 },
            fixup: cubek_microkernel::attention::packed_cache::KeyFixup::RoundOnly,
        };
        let packed = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Packed(layout),
            ..qwen(1, 4096)
        };
        assert!(reads(
            HypercubeLastArrivalShuffleAttentionRoutine::new(),
            &hw,
            &packed
        ));
        assert!(reads(
            HypercubeSecondLaunchShuffleAttentionRoutine::new(),
            &hw,
            &packed
        ));
        declines(UnitAttentionRoutine, &hw, &packed);
        declines(PlaneFragmentAttentionRoutine, &hw, &packed);
        declines(
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &hw,
            &packed,
        );
        declines(
            CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new(),
            &hw,
            &packed,
        );
        declines(CubeShuffleAttentionRoutine, &hw, &packed);
        declines(HypercubeSecondLaunchFragmentAttentionRoutine, &hw, &packed);
    }

    fn assert_steps_divide<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
        routine: R,
        hw: &Hardware,
    ) -> usize {
        let problems = [qwen(512, 512), qwen(1, 4096)];
        let mut prepared = 0;
        for settings in routine.settings(hw) {
            for problem in &problems {
                let Ok((blueprint, _)) = routine.prepare(hw, problem, &settings) else {
                    continue;
                };
                prepared += 1;
                let step = blueprint.partitioning.stage().extent(S);
                assert!(
                    AttentionRoutineProblem::WIDEST_STEP.is_multiple_of(step),
                    "{settings} on {hw:?} steps {step} keys"
                );
            }
        }
        prepared
    }

    /// A single position reads its group as rows of one head, every one attending every key: a
    /// causal single position sees every attended key, so nothing is masked differently. More
    /// positions keep their own rows and visibility.
    #[test]
    fn a_single_position_reads_its_group_as_rows() {
        assert_eq!(
            QueryAxis::new(1).read(4, 1, AttentionVisibility::Causal),
            (1, 4, AttentionVisibility::Attended)
        );
        assert_eq!(
            QueryAxis::new(7).read(4, 7, AttentionVisibility::Causal),
            (4, 7, AttentionVisibility::Causal)
        );
    }

    /// A windowed single position keeps its window as rows: every head of the group sees the
    /// last keys the window spans, which no longer depends on which row it is. More positions
    /// keep their own window.
    #[test]
    fn a_single_position_keeps_its_window_as_rows() {
        let window = core::num::NonZeroUsize::new(512).unwrap();
        let windowed = AttentionVisibility::new(true, Some(window));
        assert_eq!(windowed, AttentionVisibility::Windowed { window });
        assert_eq!(
            QueryAxis::new(1).read(4, 1, windowed),
            (1, 4, AttentionVisibility::Recent { window })
        );
        assert_eq!(QueryAxis::new(7).read(4, 7, windowed), (4, 7, windowed));
        assert!(windowed.is_causal());
        assert!(!AttentionVisibility::Recent { window }.is_causal());
    }
}
