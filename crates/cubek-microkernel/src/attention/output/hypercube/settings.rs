use crate::attention::CubeAttentionStageLayout;
use crate::attention::output::cube::CubeAttentionMicrokernelSettings;
use crate::attention::output::plane::{
    PlaneFragmentAttentionMicrokernelSettings, PlaneShuffleAttentionMicrokernelSettings,
};
use crate::matmul::dense::plane::PlaneFragmentTransport;
use crate::staging::StageRing;
use core::num::NonZeroUsize;
use cubek_std::Hardware;

/// The settings of the cube every cube of the hypercube runs, over planes of the plane
/// microkernel `P`: its own walk of its run of the keys is a cube's ([`CubeAttentionMicrokernel`](crate::attention::output::cube::CubeAttentionMicrokernel)),
/// with every knob a cube has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAttentionMicrokernelSettings<P> {
    pub cube: CubeAttentionMicrokernelSettings<P>,
}

impl HypercubeAttentionMicrokernelSettings<PlaneShuffleAttentionMicrokernelSettings> {
    /// The plane shuffle's default, a plane of one head a cube.
    pub const DEFAULT: Self = Self {
        cube: CubeAttentionMicrokernelSettings {
            plane: PlaneShuffleAttentionMicrokernelSettings::DEFAULT,
            heads: NonZeroUsize::MIN,
            planes_along_queries: NonZeroUsize::MIN,
            planes_along_keys: NonZeroUsize::MIN,
            ring: StageRing::SINGLE,
            layout: CubeAttentionStageLayout::Strided,
        },
    };
}

impl HypercubeAttentionMicrokernelSettings<PlaneFragmentAttentionMicrokernelSettings> {
    /// The cube's plane fragments' default, a plane of one head a cube, its planes reading the
    /// score through their windows: what every device's fragments run.
    pub const DEFAULT: Self = Self {
        cube:
            CubeAttentionMicrokernelSettings {
                plane:
                    PlaneFragmentAttentionMicrokernelSettings {
                        transport: PlaneFragmentTransport::Cmma,
                        ..CubeAttentionMicrokernelSettings::<
                            PlaneFragmentAttentionMicrokernelSettings,
                        >::DEFAULT
                            .plane
                    },
                heads: NonZeroUsize::MIN,
                planes_along_queries: NonZeroUsize::MIN,
                planes_along_keys: NonZeroUsize::MIN,
                ring: StageRing::SINGLE,
                layout: CubeAttentionStageLayout::Strided,
            },
    };
}

/// The settings of a plane microkernel a hypercube's cubes hold, and the hypercube's settings worth
/// trying over it: its knobs are worth different values over each plane microkernel, so its list
/// is one for each.
pub trait HypercubeAttentionPlaneSettings: Sized {
    /// The hypercube's settings worth trying on this device over planes of this kind, its default
    /// first where it is listed.
    fn hypercube_settings(hw: &Hardware) -> Vec<HypercubeAttentionMicrokernelSettings<Self>>;
}

impl HypercubeAttentionPlaneSettings for PlaneShuffleAttentionMicrokernelSettings {
    /// Every one of two, four and eight rows a plane, eight and sixteen units on a key and two and
    /// four keys in flight, a plane a cube; then four planes along a cube's run of the keys over the
    /// same units and keys at two rows, four and eight planes at four rows, and four planes at
    /// eight rows.
    ///
    /// A decode step's rows are its head group: two, four or eight query heads to a key head. A
    /// plane of four rows on a group of two spends half its arithmetic on rows that hold no query.
    ///
    /// Planes along the keys put more reads in flight on a cube without adding slots to merge: a
    /// cube's planes merge into its one slot before it is stored.
    fn hypercube_settings(_hw: &Hardware) -> Vec<HypercubeAttentionMicrokernelSettings<Self>> {
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let plane =
            |rows, units_per_key, keys_in_flight| PlaneShuffleAttentionMicrokernelSettings {
                seq_q: count(rows),
                units_per_key: count(units_per_key),
                keys_in_flight: count(keys_in_flight),
            };
        let mut layouts = Vec::new();
        for (rows, planes_along_keys) in [(2, 1), (4, 1), (8, 1), (2, 4), (4, 4), (4, 8)] {
            for units_per_key in [8, 16] {
                for keys_in_flight in [2, 4] {
                    layouts.push((
                        plane(rows, units_per_key, keys_in_flight),
                        planes_along_keys,
                    ));
                }
            }
        }
        layouts.push((plane(8, 16, 2), 4));
        let lead = HypercubeAttentionMicrokernelSettings::<Self>::DEFAULT.cube;
        layouts
            .into_iter()
            .map(
                |(plane, planes_along_keys)| HypercubeAttentionMicrokernelSettings {
                    cube: CubeAttentionMicrokernelSettings {
                        plane,
                        planes_along_keys: count(planes_along_keys),
                        ..lead
                    },
                },
            )
            .collect()
    }
}

impl HypercubeAttentionPlaneSettings for PlaneFragmentAttentionMicrokernelSettings {
    /// Its default, then the cube's swizzled boxes of fragments over its heads, the score kept in
    /// the fragments: a cube of the cube routine's at its widest, on a run of the keys; then the
    /// one-by-eight box on two planes along a head's queries. Every one holds one plane along its
    /// run, as a staging cube does; the swizzled ones decline where the device reads the score
    /// through windows alone.
    fn hypercube_settings(_hw: &Hardware) -> Vec<HypercubeAttentionMicrokernelSettings<Self>> {
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let cube = CubeAttentionMicrokernelSettings::<Self>::DEFAULT;
        let swizzled =
            [(1, 8), (2, 4), (2, 8)].map(|(rows, keys)| HypercubeAttentionMicrokernelSettings {
                cube: CubeAttentionMicrokernelSettings {
                    plane: PlaneFragmentAttentionMicrokernelSettings {
                        seq_q: count(rows),
                        seq_kv: count(keys),
                        ..cube.plane
                    },
                    layout: CubeAttentionStageLayout::SwizzledBlocks,
                    ..cube
                },
            });
        // A prompt of few heads leaves the device short of cubes. Two planes along the queries,
        // the keys split four ways across cubes, take Qwen3-0.6B's and Qwen3-1.7B's prompt of 512
        // on an RTX PRO 6000 to 43 µs, from the cube routine's 49 on the same planes and 74 on one.
        let split = HypercubeAttentionMicrokernelSettings {
            cube: CubeAttentionMicrokernelSettings {
                planes_along_queries: count(2),
                ..swizzled[0].cube
            },
        };
        [HypercubeAttentionMicrokernelSettings::<Self>::DEFAULT]
            .into_iter()
            .chain(swizzled)
            .chain([split])
            .collect()
    }
}

/// Its cube's plane, then what of the cube is not a plane of one head: the heads, the planes along
/// a head's queries and along its keys, the ring and the layout.
impl<P: core::fmt::Display> core::fmt::Display for HypercubeAttentionMicrokernelSettings<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let cube = &self.cube;
        write!(f, "{}", cube.plane)?;
        for (name, count) in [
            ("heads", cube.heads),
            ("planes_along_queries", cube.planes_along_queries),
            ("planes_along_keys", cube.planes_along_keys),
        ] {
            if count.get() > 1 {
                write!(f, " {name}={count}")?;
            }
        }
        write!(f, "{}{}", cube.ring, cube.layout)
    }
}
