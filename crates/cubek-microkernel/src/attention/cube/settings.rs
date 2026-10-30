use crate::attention::cube::CubeAttentionPlaneSettings;
use crate::attention::plane::{
    InstructionShape, PlaneFragmentAttentionMicrokernelSettings,
    PlaneShuffleAttentionMicrokernelSettings,
};
use crate::matmul::cube::CubeStagedDelivery;
use crate::matmul::plane::{PlaneFragmentDepth, PlaneFragmentTransport};
use core::num::NonZeroUsize;
use cubecl::tile::Scratch;
use cubek_std::microkernel::{StageLayout, StageRing};

/// The settings of the plane microkernel on each of its planes, the heads a cube holds and the
/// planes each head's queries and keys are laid over, and, where its planes read a stage, how the
/// cube holds its stages of keys and values: how many at once, where the next one waits, and how
/// each is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeAttentionMicrokernelSettings {
    pub plane: CubeAttentionPlaneSettings,
    /// Heads of the group one cube holds, which read the keys it stages.
    pub heads: NonZeroUsize,
    /// Planes along one head's queries.
    pub planes_along_queries: NonZeroUsize,
    /// Planes along the keys of the same rows, each taking its block of every stage.
    pub planes_along_keys: NonZeroUsize,
    /// How it holds its stages of keys and values.
    pub ring: StageRing,
    /// How each of them is laid out in shared memory.
    pub layout: StageLayout,
    /// Who moves each of them out of global memory: the cube's units, or the copy engine their
    /// lines are handed to. A request the microkernel resolves against the device, its ring and
    /// the lines' widths, so every setting runs everywhere.
    pub delivery: CubeStagedDelivery,
}

impl CubeAttentionMicrokernelSettings {
    /// The lead over plane fragments: a fragment of rows and one of keys a plane, four heads a
    /// cube, what a cube holds for a head dim of 128 in `f32` within 32 KiB of shared memory.
    pub const FRAGMENT: Self = Self {
        plane: CubeAttentionPlaneSettings::Fragment(
            PlaneFragmentAttentionMicrokernelSettings::fragments(1, 1),
        ),
        heads: NonZeroUsize::new(4).unwrap(),
        planes_along_queries: NonZeroUsize::new(1).unwrap(),
        planes_along_keys: NonZeroUsize::new(1).unwrap(),
        ring: StageRing::SINGLE,
        layout: StageLayout::Strided,
        delivery: CubeStagedDelivery::Copy,
    };

    /// The lead over plane shuffles: four planes of the plane shuffle's default splitting one
    /// head's keys, a cube's worth of planes on every box of rows, where the shuffle alone would
    /// leave a cube one plane.
    pub const SHUFFLE: Self = Self {
        plane: CubeAttentionPlaneSettings::Shuffle(
            PlaneShuffleAttentionMicrokernelSettings::DEFAULT,
        ),
        heads: NonZeroUsize::new(1).unwrap(),
        planes_along_queries: NonZeroUsize::new(1).unwrap(),
        planes_along_keys: NonZeroUsize::new(4).unwrap(),
        ring: StageRing::SINGLE,
        layout: StageLayout::Strided,
        delivery: CubeStagedDelivery::Copy,
    };

    /// These settings with their plane fragments holding the score the whole head dim deep
    /// ([`PlaneFragmentDepth::Stage`]); settings over plane shuffles, which hold no score
    /// fragments, as they are.
    pub fn whole_head(self) -> Self {
        match self.plane {
            CubeAttentionPlaneSettings::Fragment(plane) => Self {
                plane: CubeAttentionPlaneSettings::Fragment(
                    PlaneFragmentAttentionMicrokernelSettings {
                        score_depth: PlaneFragmentDepth::Stage,
                        ..plane
                    },
                ),
                ..self
            },
            CubeAttentionPlaneSettings::Shuffle(_) => self,
        }
    }

    /// The cube's settings worth trying over planes of the kind this lead runs, the lead first: a
    /// cube's own knobs are worth different values over each plane microkernel, so its list is one
    /// for each.
    pub fn listed(self) -> Vec<Self> {
        match self.plane {
            CubeAttentionPlaneSettings::Fragment(plane) => Self::fragments_listed(self, plane),
            CubeAttentionPlaneSettings::Shuffle(plane) => Self::shuffles_listed(self, plane),
        }
    }

    /// A star off `lead`: each larger box of fragments, each count of heads, two and four planes
    /// along a head's queries, each other ring of stages and layout of a stage, the whole grid
    /// bounced at once, a rectangular cmma instruction read through the windows, and one head a
    /// cube with eight and sixteen fragments along its keys, or with two, four and eight planes
    /// along them, so a step reaches the hundreds of keys, with the default's others; then the
    /// larger boxes again in swizzled tiles, which a plane keeping its score in fragments reads
    /// without bank conflicts and a cmma plane declines, the one-by-eight box among them also on
    /// two planes along a head's queries, and the one-by-two box on two heads and two planes along
    /// their queries; then the lead's box with its score the whole head dim deep; then the
    /// one-by-four box on two slots the copy engine
    /// fills, over two and four heads. The wide steps hold one head because a step's stages and the
    /// heads' windows share one budget; one that still does not fit the device's shared memory
    /// declines there.
    fn fragments_listed(lead: Self, plane: PlaneFragmentAttentionMicrokernelSettings) -> Vec<Self> {
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let with = move |plane| Self {
            plane: CubeAttentionPlaneSettings::Fragment(plane),
            ..lead
        };
        let fragments = move |rows: usize, keys: usize| {
            with(PlaneFragmentAttentionMicrokernelSettings {
                seq_q: count(rows),
                seq_kv: count(keys),
                ..plane
            })
        };
        let heads = [1, 2, 8].map(|heads| Self {
            heads: count(heads),
            ..lead
        });
        let split = [2, 4].map(|planes| Self {
            planes_along_queries: count(planes),
            ..lead
        });
        // One head a cube, so the sum of the planes' products fits beside the stages, and the
        // score read through the windows, so the planes' states merge row by row.
        let along_keys = [2, 4, 8].map(|planes| Self {
            plane: CubeAttentionPlaneSettings::Fragment(
                PlaneFragmentAttentionMicrokernelSettings {
                    transport: PlaneFragmentTransport::Cmma,
                    ..plane
                },
            ),
            heads: count(1),
            planes_along_keys: count(planes),
            ..lead
        });
        let rings = [
            StageRing::DOUBLE,
            StageRing::SINGLE_THROUGH_REGISTERS,
            StageRing::DOUBLE_THROUGH_REGISTERS,
        ]
        .map(|ring| Self { ring, ..lead });
        let layouts = [
            StageLayout::PaddedRows,
            StageLayout::Blocks,
            StageLayout::PaddedBlocks,
            StageLayout::SwizzledBlocks,
        ]
        .map(|layout| Self { layout, ..lead });
        let whole = Self {
            plane: CubeAttentionPlaneSettings::Fragment(
                PlaneFragmentAttentionMicrokernelSettings {
                    scratch: Scratch::WholeGrid,
                    ..plane
                },
            ),
            ..lead
        };
        // A rectangular cmma instruction, read through the windows: the only one a device offers
        // for some elements, as tf32 for f32 operands on NVIDIA.
        let rectangular = Self {
            plane: CubeAttentionPlaneSettings::Fragment(
                PlaneFragmentAttentionMicrokernelSettings {
                    transport: PlaneFragmentTransport::Cmma,
                    shape: InstructionShape::Rectangular,
                    ..plane
                },
            ),
            ..lead
        };
        let wide = [8, 16].map(|keys| Self {
            plane: CubeAttentionPlaneSettings::Fragment(
                PlaneFragmentAttentionMicrokernelSettings {
                    seq_kv: count(keys),
                    ..plane
                },
            ),
            heads: count(1),
            ..lead
        });
        let swizzled = [(2, 4), (2, 8), (1, 8)].map(|(rows, keys)| Self {
            layout: StageLayout::SwizzledBlocks,
            ..fragments(rows, keys)
        });
        // A prompt of few heads leaves the device short of cubes, one a box of queries for every
        // four heads: more planes on each box are what it has left to run. On Qwen3-0.6B's prompt
        // of 512 on an RTX PRO 6000, two planes along the queries take the one-by-eight box from
        // 74 to 49 µs; on Qwen3-8B, whose cubes already fill the device, they cost 4%. Four were
        // slower than two on both.
        let swizzled_split = Self {
            planes_along_queries: count(2),
            ..swizzled[2]
        };
        // Four planes a cube sharing each stage, two heads over two boxes of queries, each plane a
        // one-by-two box: thirty-two rows over sixteen keys a step, the work a step of the attention
        // Apple ships for Metal does. On an M2 Pro (Qwen3-0.6B's 16/8 heads, and Qwen3-4B's 32/8)
        // it is the fastest cube setting, 5 to 16% under the lead taking its queries last first, at
        // 512 and 2048 queries.
        let four_planes_split = Self {
            heads: count(2),
            planes_along_queries: count(2),
            layout: StageLayout::SwizzledBlocks,
            ..fragments(1, 2)
        };
        // The score's fragments the whole head dim deep: one contraction a stage of keys instead of
        // a walk of the head dim's steps. On an M2 Pro (Qwen3-0.6B's 16/8 heads and Qwen3-4B's
        // 32/8), taken last first as the lead is, it is 1.5 to 4% under the lead at 512 and 2048
        // queries.
        let whole_head = lead.whole_head();
        // The copy engine lands a stage while the planes contract the one before, so it wants a
        // second slot. On Qwen3-8B's last 8k prompt chunk on an RTX PRO 6000 (32/8 heads, 512
        // queries over 8192 keys), the one-by-four box on two slots takes 470 µs with the units
        // copying and 406 to 411 with the engine; on one slot the engine cost 2%.
        let copied = [
            (2, 4, StageLayout::SwizzledBlocks),
            (4, 2, StageLayout::PaddedBlocks),
        ]
        .map(|(heads, planes, layout)| Self {
            heads: count(heads),
            planes_along_queries: count(planes),
            ring: StageRing::DOUBLE,
            layout,
            delivery: CubeStagedDelivery::AsyncCopy,
            ..fragments(1, 4)
        });
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
            .chain([swizzled_split, four_planes_split, whole_head])
            .chain(copied)
            .collect()
    }

    /// A star off `lead`: two and eight planes along a head's keys, eight rows a plane,
    /// thirty-two units on a key, four keys in flight, four heads a cube, and two planes along a
    /// head's queries.
    fn shuffles_listed(lead: Self, plane: PlaneShuffleAttentionMicrokernelSettings) -> Vec<Self> {
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let along_keys = [2, 8].map(|planes| Self {
            planes_along_keys: count(planes),
            ..lead
        });
        let with = |shuffle| Self {
            plane: CubeAttentionPlaneSettings::Shuffle(shuffle),
            ..lead
        };
        let rows = with(PlaneShuffleAttentionMicrokernelSettings {
            seq_q: count(8),
            ..plane
        });
        let units = with(PlaneShuffleAttentionMicrokernelSettings {
            units_per_key: count(32),
            ..plane
        });
        let in_flight = with(PlaneShuffleAttentionMicrokernelSettings {
            keys_in_flight: count(4),
            ..plane
        });
        let heads = Self {
            heads: count(4),
            ..lead
        };
        let split = Self {
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

impl core::fmt::Display for CubeAttentionMicrokernelSettings {
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
        )?;
        match self.delivery {
            CubeStagedDelivery::Copy => Ok(()),
            CubeStagedDelivery::AsyncCopy => write!(f, " delivery=async"),
            CubeStagedDelivery::Tma => write!(f, " delivery=tma"),
        }
    }
}
