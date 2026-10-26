use crate::attention::CubeAttentionStageLayout;
use crate::attention::output::plane::{
    InstructionShape, PlaneFragmentAttentionMicrokernelSettings,
    PlaneShuffleAttentionMicrokernelSettings,
};
use crate::attention::output::unit::UnitAttentionMicrokernelSettings;
use crate::matmul::dense::plane::PlaneFragmentTransport;
use crate::staging::StageRing;
use core::num::NonZeroUsize;
use cubecl::tile::Scratch;
use cubek_std::Hardware;

/// The plane microkernel's settings `P`, the heads a cube holds and the planes each head's queries
/// and keys are laid over, and, where the plane microkernel reads a stage, how the cube holds its
/// stages of keys and values: how many at once, where the next one waits, and how each is laid
/// out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeAttentionMicrokernelSettings<P> {
    pub plane: P,
    /// Heads of the group one cube holds, which read the keys it stages.
    pub heads: NonZeroUsize,
    /// Planes along one head's queries.
    pub planes_along_queries: NonZeroUsize,
    /// Planes along the keys of the same rows, each taking its block of every stage.
    pub planes_along_keys: NonZeroUsize,
    /// How it holds its stages of keys and values.
    pub ring: StageRing,
    /// How each of them is laid out in shared memory.
    pub layout: CubeAttentionStageLayout,
}

impl CubeAttentionMicrokernelSettings<PlaneFragmentAttentionMicrokernelSettings> {
    /// A fragment of rows and one of keys a plane, four heads a cube: what a cube holds for a head
    /// dim of 128 in `f32` within 32 KiB of shared memory.
    pub const DEFAULT: Self = Self {
        plane: PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1),
        heads: NonZeroUsize::new(4).unwrap(),
        planes_along_queries: NonZeroUsize::new(1).unwrap(),
        planes_along_keys: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
        layout: CubeAttentionStageLayout::Strided,
    };
}

impl CubeAttentionMicrokernelSettings<PlaneShuffleAttentionMicrokernelSettings> {
    /// Four planes of the plane shuffle's default splitting one head's keys: a cube's worth of planes on
    /// every box of rows, where the shuffle alone would leave a cube one plane.
    pub const DEFAULT: Self = Self {
        plane: PlaneShuffleAttentionMicrokernelSettings::DEFAULT,
        heads: NonZeroUsize::new(1).unwrap(),
        planes_along_queries: NonZeroUsize::new(1).unwrap(),
        planes_along_keys: NonZeroUsize::new(4).unwrap(),
        ring: StageRing::SINGLE,
        layout: CubeAttentionStageLayout::Strided,
    };
}

impl CubeAttentionMicrokernelSettings<UnitAttentionMicrokernelSettings> {
    /// A plane of the unit microkernel's default, a row a unit, for one head: a stage of keys
    /// read by a plane's worth of rows.
    pub const DEFAULT: Self = Self {
        plane: UnitAttentionMicrokernelSettings::DEFAULT,
        heads: NonZeroUsize::new(1).unwrap(),
        planes_along_queries: NonZeroUsize::new(1).unwrap(),
        planes_along_keys: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
        layout: CubeAttentionStageLayout::Strided,
    };
}

/// The settings of a plane microkernel a cube holds, and the cube's settings worth trying over it:
/// a cube's own knobs are worth different values over each plane microkernel, so its list is one
/// for each.
pub trait CubeAttentionPlaneSettings: Sized {
    /// The cube's settings worth trying on this device over planes of this kind, its default
    /// first.
    fn cube_settings(hw: &Hardware) -> Vec<CubeAttentionMicrokernelSettings<Self>>;
}

impl CubeAttentionPlaneSettings for PlaneFragmentAttentionMicrokernelSettings {
    /// A star off the default: each larger box of fragments, each count of heads, two and four
    /// planes along a head's queries, each other ring of stages and layout of a stage, the whole
    /// grid bounced at once, a rectangular cmma instruction read through the windows, and one head
    /// a cube with eight and sixteen fragments along its keys, or with two, four and eight planes
    /// along them, so a step reaches the hundreds of keys, with the default's others; then the
    /// larger boxes again in swizzled tiles, which a plane keeping its score in fragments reads
    /// without bank conflicts and a cmma plane declines, the one-by-eight box among them also on two
    /// planes along a head's queries. The wide steps hold one head because a step's
    /// stages and the heads' windows share one budget; one that still does not fit the device's
    /// shared memory declines there.
    fn cube_settings(_hw: &Hardware) -> Vec<CubeAttentionMicrokernelSettings<Self>> {
        let lead = CubeAttentionMicrokernelSettings::<Self>::DEFAULT;
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let fragments = move |rows: usize, keys: usize| CubeAttentionMicrokernelSettings {
            plane: PlaneFragmentAttentionMicrokernelSettings {
                seq_q: count(rows),
                seq_kv: count(keys),
                ..lead.plane
            },
            ..lead
        };
        let heads = [1, 2, 8].map(|heads| CubeAttentionMicrokernelSettings {
            heads: count(heads),
            ..lead
        });
        let split = [2, 4].map(|planes| CubeAttentionMicrokernelSettings {
            planes_along_queries: count(planes),
            ..lead
        });
        // One head a cube, so the sum of the planes' products fits beside the stages, and the
        // score read through the windows, so the planes' states merge row by row.
        let along_keys = [2, 4, 8].map(|planes| CubeAttentionMicrokernelSettings {
            plane: PlaneFragmentAttentionMicrokernelSettings {
                transport: PlaneFragmentTransport::Cmma,
                ..lead.plane
            },
            heads: count(1),
            planes_along_keys: count(planes),
            ..lead
        });
        let rings = [
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
            StageRing::DOUBLE_THROUGH_REGISTERS,
        ]
        .map(|ring| CubeAttentionMicrokernelSettings { ring, ..lead });
        let layouts = [
            CubeAttentionStageLayout::PaddedRows,
            CubeAttentionStageLayout::Blocks,
            CubeAttentionStageLayout::PaddedBlocks,
            CubeAttentionStageLayout::SwizzledBlocks,
        ]
        .map(|layout| CubeAttentionMicrokernelSettings { layout, ..lead });
        let whole = CubeAttentionMicrokernelSettings {
            plane: PlaneFragmentAttentionMicrokernelSettings {
                scratch: Scratch::WholeGrid,
                ..lead.plane
            },
            ..lead
        };
        // A rectangular cmma instruction, read through the windows: the only one a device offers
        // for some elements, as tf32 for f32 operands on NVIDIA.
        let rectangular = CubeAttentionMicrokernelSettings {
            plane: PlaneFragmentAttentionMicrokernelSettings {
                transport: PlaneFragmentTransport::Cmma,
                shape: InstructionShape::Rectangular,
                ..lead.plane
            },
            ..lead
        };
        let wide = [8, 16].map(|keys| CubeAttentionMicrokernelSettings {
            plane: PlaneFragmentAttentionMicrokernelSettings {
                seq_kv: count(keys),
                ..lead.plane
            },
            heads: count(1),
            ..lead
        });
        let swizzled =
            [(2, 4), (2, 8), (1, 8)].map(|(rows, keys)| CubeAttentionMicrokernelSettings {
                layout: CubeAttentionStageLayout::SwizzledBlocks,
                ..fragments(rows, keys)
            });
        // A prompt of few heads leaves the device short of cubes, one a box of queries for every
        // four heads: more planes on each box are what it has left to run. On Qwen3-0.6B's prompt
        // of 512 on an RTX PRO 6000, two planes along the queries take the one-by-eight box from
        // 74 to 49 µs; on Qwen3-8B, whose cubes already fill the device, they cost 4%. Four were
        // slower than two on both.
        let swizzled_split = CubeAttentionMicrokernelSettings {
            planes_along_queries: count(2),
            ..swizzled[2]
        };
        [lead, fragments(2, 2), fragments(2, 4)]
            .into_iter()
            .chain(wide)
            .chain(heads)
            .chain(split)
            .chain(rings)
            .chain(layouts)
            .chain([whole, rectangular])
            .chain(along_keys)
            .chain(swizzled)
            .chain([swizzled_split])
            .collect()
    }
}

impl CubeAttentionPlaneSettings for PlaneShuffleAttentionMicrokernelSettings {
    /// A star off the default: two and eight planes along a head's keys, eight rows a plane,
    /// thirty-two units on a key, four keys in flight, four heads a cube, and two planes along a
    /// head's queries.
    fn cube_settings(_hw: &Hardware) -> Vec<CubeAttentionMicrokernelSettings<Self>> {
        let lead = CubeAttentionMicrokernelSettings::<Self>::DEFAULT;
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let along_keys = [2, 8].map(|planes| CubeAttentionMicrokernelSettings {
            planes_along_keys: count(planes),
            ..lead
        });
        let plane = |plane| CubeAttentionMicrokernelSettings { plane, ..lead };
        let rows = plane(PlaneShuffleAttentionMicrokernelSettings {
            seq_q: count(8),
            ..lead.plane
        });
        let units = plane(PlaneShuffleAttentionMicrokernelSettings {
            units_per_key: count(32),
            ..lead.plane
        });
        let in_flight = plane(PlaneShuffleAttentionMicrokernelSettings {
            keys_in_flight: count(4),
            ..lead.plane
        });
        let heads = CubeAttentionMicrokernelSettings {
            heads: count(4),
            ..lead
        };
        let split = CubeAttentionMicrokernelSettings {
            planes_along_queries: count(2),
            ..lead
        };
        [lead]
            .into_iter()
            .chain(along_keys)
            .chain([rows, units, in_flight, heads, split])
            .collect()
    }
}

impl CubeAttentionPlaneSettings for UnitAttentionMicrokernelSettings {
    /// A star off the default, a plane of the unit microkernel's default, a row a unit, eight keys
    /// a stage: sixteen and thirty-two keys a stage, two rows a unit, two heads a cube, two planes
    /// along a head's queries, and a second stage in flight.
    fn cube_settings(_hw: &Hardware) -> Vec<CubeAttentionMicrokernelSettings<Self>> {
        let lead = CubeAttentionMicrokernelSettings::<Self>::DEFAULT;
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let unit = |rows: usize, keys: usize| CubeAttentionMicrokernelSettings {
            plane: UnitAttentionMicrokernelSettings {
                seq_q: count(rows),
                seq_kv: count(keys),
            },
            ..lead
        };
        let heads = CubeAttentionMicrokernelSettings {
            heads: count(2),
            ..lead
        };
        let split = CubeAttentionMicrokernelSettings {
            planes_along_queries: count(2),
            ..lead
        };
        let ahead = CubeAttentionMicrokernelSettings {
            ring: StageRing::DOUBLE,
            ..lead
        };
        vec![
            lead,
            unit(1, 16),
            unit(1, 32),
            unit(2, 8),
            heads,
            split,
            ahead,
        ]
    }
}

impl<P: core::fmt::Display> core::fmt::Display for CubeAttentionMicrokernelSettings<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} heads={} planes_along_queries={} planes_along_keys={}{}{}",
            self.plane,
            self.heads,
            self.planes_along_queries,
            self.planes_along_keys,
            self.ring,
            self.layout
        )
    }
}
