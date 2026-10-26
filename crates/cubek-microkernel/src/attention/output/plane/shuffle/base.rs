use crate::attention::CubeAttentionStageLayout;
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionOperands, AttentionPartitioning,
    AttentionStagingCount, AttentionWorkItem, CubePlaneKeys, CubePlaneLayout, CubePlaneMicrokernel,
    CubePlaneReads, G, Q, S,
};
use crate::matmul::MatmulOrder;
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A few rows of one head a plane owns in its units' registers, the head and value dims split
/// between the units: each step of keys is read straight from where it lies, scored, taken into
/// the rows' softmax and summed against its values with no shared memory and no barrier.
///
/// What few rows against many keys want, a decode step's above all: the rows are too few to fill
/// a matrix instruction, and each key is read by one plane only, so staging it buys nothing. The
/// plane's units form teams of [`units_per_key`](PlaneShuffleAttentionMicrokernelSettings) units,
/// each team scoring its own keys of every step: a unit holds every `units_per_key`-th line of
/// each row's query and output, a key's score is its team's partial dots summed across the team,
/// and every team keeps its own softmax state until `close` merges them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleAttentionMicrokernel {
    pub settings: PlaneShuffleAttentionMicrokernelSettings,
    /// Teams a plane holds: the device's plane width over the units each team puts on a key, the
    /// stride between one team's keys of a step. A count a blueprint holds, an exception to the
    /// crate's rule: the teams' partials are summed across exactly this many lanes.
    pub teams: usize,
}

/// The rows a plane holds, how many of its units share one key, and how many keys each team
/// reads at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleAttentionMicrokernelSettings {
    /// Query rows of one head, each held whole in the plane's registers.
    pub seq_q: NonZeroUsize,
    /// Units scoring one key together, each holding its share of the head and value dims: fewer
    /// put more teams beside each other, more hold fewer lines each.
    pub units_per_key: NonZeroUsize,
    /// Keys one team loads before it uses any, so their reads are in flight together.
    pub keys_in_flight: NonZeroUsize,
}

impl PlaneShuffleAttentionMicrokernelSettings {
    /// A decode step's group of four read as rows, two teams of a 32-wide plane, two keys each.
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::new(4).unwrap(),
        units_per_key: NonZeroUsize::new(16).unwrap(),
        keys_in_flight: NonZeroUsize::new(2).unwrap(),
    };
}

impl core::fmt::Display for PlaneShuffleAttentionMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "shuffle rows={} units_per_key={} keys_in_flight={}",
            self.seq_q, self.units_per_key, self.keys_in_flight
        )
    }
}

impl PlaneShuffleAttentionMicrokernel {
    /// The most rows a plane holds: each row's query and output lines sit in every unit's
    /// registers, so past this many the registers, not the keys, bound the kernel.
    pub const MAX_ROWS: usize = 8;

    /// Units the plane holds: its teams, each of its units per key.
    pub fn units(&self) -> usize {
        self.teams * self.settings.units_per_key.get()
    }

    /// 32-bit registers one unit keeps: its lines of every row's query and output and their
    /// softmax state, and the lines of each key in flight. The compiler adds addressing on top.
    fn registers_per_unit(&self, item: &AttentionWorkItem) -> u32 {
        let units_per_key = self.settings.units_per_key.get();
        let held = self.seq_q() * ((item.head_dim + item.val_dim).div_ceil(units_per_key) + 2);
        let in_flight = self.settings.keys_in_flight.get()
            * (item.head_dim + item.val_dim).div_ceil(units_per_key);
        (held * item.sum.size() + in_flight * item.elements.key.size()).div_ceil(4) as u32
    }
}

/// Its settings.
impl core::fmt::Display for PlaneShuffleAttentionMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for PlaneShuffleAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = PlaneShuffleAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = "the plane shuffle sums a key's partial dots across its team, \
         which needs a plane width the device commits to and its settings' units per key divide, \
         a power of two, and holds at most `PlaneShuffleAttentionMicrokernel::MAX_ROWS` rows";

    /// A star off the default: eight rows a plane, thirty-two units on a key, four keys in flight.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        let lead = Self::Settings::DEFAULT;
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        vec![
            lead,
            Self::Settings {
                seq_q: count(8),
                ..lead
            },
            Self::Settings {
                units_per_key: count(32),
                ..lead
            },
            Self::Settings {
                keys_in_flight: count(4),
                ..lead
            },
        ]
    }

    /// The one its settings size, where the device commits to a plane width that the units per
    /// key divide, its teams summing their partials across exactly that many lanes, and its rows
    /// are at most [`MAX_ROWS`](Self::MAX_ROWS). The width is a power of two, and so the units
    /// per key that divide it: a team is the lanes a mask of the low bits groups.
    fn candidates(
        hw: &Hardware,
        _item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        let width = hw.plane_dim().ok().map(|width| width as usize);
        width
            .filter(|width| {
                width.is_power_of_two()
                    && width.is_multiple_of(settings.units_per_key.get())
                    && settings.seq_q.get() <= Self::MAX_ROWS
            })
            .map(|width| Self {
                settings,
                teams: width / settings.units_per_key.get(),
            })
            .into_iter()
    }

    /// The query, keys and values lined alike: a unit's line of a row's query meets the same
    /// line of each key, and its line of the row's output sums the same line of each value.
    fn vectorize(
        &self,
        _partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        for operand in [
            AttentionOperand::Key,
            AttentionOperand::Value,
            AttentionOperand::Out,
        ] {
            vectorization.require(VectorRequirement::Together {
                operand,
                with: AttentionOperand::Query,
            });
        }
    }

    /// Its plane's reads, each key and value once for the plane's rows, a step in flight; and its
    /// units' work, in registers on the device's vector arithmetic.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        AttentionStagingCount {
            seq_kv: self.seq_kv(),
            stages_in_flight: 1,
            runs: AttentionOperands {
                query: item.head_dim,
                key: item.head_dim,
                value: item.val_dim,
                out: item.val_dim,
            },
            readers: AttentionOperands {
                query: 1,
                key: 1,
                value: 1,
                out: 1,
            },
        }
        .count(item, partitioning, estimate);
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: self.registers_per_unit(item),
        });
    }
}

impl AttentionMicrokernel for PlaneShuffleAttentionMicrokernel {
    /// A unit reads a key's lines along the head dim, as the keys lie.
    fn key_order(&self) -> MatmulOrder {
        MatmulOrder::InBufferOrder
    }
}

impl CubePlaneMicrokernel for PlaneShuffleAttentionMicrokernel {
    const NAME: &'static str = "shuffle";

    /// Each team reads its keys of every step where they lie.
    const READS: CubePlaneReads = CubePlaneReads::Global;

    /// Its planes merge in slots of shared memory.
    const KEYS: CubePlaneKeys = CubePlaneKeys::Split;

    /// Its rows of one head against a step of keys, the head and value dims whole. The plane's
    /// units lie over them in teams it places itself.
    fn tiling(&self) -> Levels {
        Levels::leaf(&[(Q, self.seq_q()), (G, 1), (S, self.seq_kv())])
    }

    fn seq_q(&self) -> usize {
        self.settings.seq_q.get()
    }

    /// Every team's keys in flight.
    fn seq_kv(&self) -> usize {
        self.teams * self.settings.keys_in_flight.get()
    }

    /// None: its rows and their keys stay in registers.
    fn window_bytes(&self, _item: &AttentionWorkItem) -> usize {
        0
    }

    /// None where a plane holds its rows alone: they leave their slots from its units'
    /// registers. Where planes split the same rows' keys, the slots they merge in first, as the
    /// cube's own rows do ([`merge_bytes`](CubePlaneMicrokernel::merge_bytes)).
    fn partial_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        self.merge_bytes(item, layout)
    }

    /// A slot for every plane of every row: a line of each element of a value, and its running
    /// max and sum ([`AttentionSlotShape`](crate::attention::AttentionSlotShape)).
    fn merge_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        match layout.planes_along_keys {
            1 => 0,
            _ => layout.planes() * self.seq_q() * (item.val_dim + 2) * item.sum.size(),
        }
    }

    /// Row-major, whatever the layout: its steps read no stage.
    fn stage_storage(&self, _layout: CubeAttentionStageLayout, _stage: &Space) -> StageStorage {
        StageStorage::Strided
    }
}
