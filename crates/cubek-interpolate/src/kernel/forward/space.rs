use super::geometry::TileGeometry;
use cubecl::client::Client;
use cubecl::{CubeCount, CubeDim};
use cubek_tile::{
    Axis, Compaction, Level, Levels, Partitioning, PhysicalAxisMap, Projection, RegisterBlock,
    Space,
};

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);

/// What a partitioning over these axes prints as ([`Partitioning::labelled`]): an [`Axis`] is an
/// index, and only the kernel that assigned it knows what it stands for.
///
/// Only tests print one today. It widens when a caller does.
#[cfg(test)]
const LABELS: [(Axis, &str); 6] = [
    (BATCH, "b"),
    (OUTPUT_H, "oh"),
    (OUTPUT_W, "ow"),
    (TAP_H, "th"),
    (TAP_W, "tw"),
    (CHANNEL, "c"),
];

/// The register block the leaf runs under, which the device decides.
pub fn register_block(client: &Client) -> RegisterBlock {
    match client.properties().hardware.num_cpu_cores {
        Some(_) => RegisterBlock::new(256).split_edge(),
        None => RegisterBlock::new(64).component_fanout(),
    }
}

/// The space one launch runs over, in the terms the kernel builds it from. The kernel's
/// comptime argument, so the space the launch sizes its grid from is the space the kernel
/// walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InterpolateSpace {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub plane_size: usize,
    pub taps: usize,
    pub geometry: TileGeometry,
}

impl InterpolateSpace {
    /// The space's axes and their extents, every one static.
    pub fn extents(&self) -> Vec<(Axis, usize)> {
        vec![
            (BATCH, self.batch),
            (OUTPUT_H, self.height),
            (OUTPUT_W, self.width),
            (TAP_H, self.taps),
            (TAP_W, self.taps),
            (CHANNEL, self.channels),
        ]
    }

    /// Four levels, outermost first: the cube grid, the channel blocks a cube walks (one region
    /// below `lanes * 4` channels), this plane's rows, then this lane's columns and channel
    /// lines.
    /// The levels, stated **from the leaf up, in counts**: one lane's columns by its channel
    /// block by a plane's rows, the lanes of a plane across columns and channels, the planes of
    /// a cube across rows, every channel block the cube's box holds, and a cube per box. The
    /// geometry was already a product — `rows_per_cube` is planes by rows, `cols_per_cube` lanes
    /// by columns — so this states the counts it holds rather than the sizes it multiplied them
    /// into. A lane split of one along an axis is no split, and the axis is handed down whole.
    pub fn levels(&self) -> Vec<Level> {
        let (geometry, plane_size) = (self.geometry, self.plane_size);
        assert!(
            geometry.lane_cols * geometry.lane_channels == plane_size,
            "InterpolateSpace: the lane split covers {} of the plane's {plane_size} lanes",
            geometry.lane_cols * geometry.lane_channels
        );
        let lanes: Vec<(Axis, usize)> = [
            (OUTPUT_W, geometry.lane_cols),
            (CHANNEL, geometry.lane_channels),
        ]
        .into_iter()
        .filter(|&(_, lanes)| lanes > 1)
        .collect();
        Levels::leaf(&[
            (OUTPUT_W, geometry.cols_per_lane),
            (CHANNEL, geometry.channel_block),
            (OUTPUT_H, geometry.rows_per_plane),
        ])
        .units(&lanes)
        .planes(&[(OUTPUT_H, geometry.planes_per_cube)])
        .walk_every(&[CHANNEL])
        .cubes(&[OUTPUT_W, OUTPUT_H])
        .batches(&[BATCH])
        .build()
    }

    pub fn space(&self) -> Space {
        Space::new(&self.extents())
    }

    /// The space with the levels that cut it: what the leaf and the overhangs are read off.
    pub fn partitioning(&self) -> Partitioning {
        Partitioning::new(self.space(), self.levels())
    }

    /// The grid this launch runs on: a cube per box of the output and per batch, the geometry's
    /// planes in each.
    pub fn grid(&self) -> (CubeCount, CubeDim) {
        let geometry = self.geometry;
        (
            CubeCount::Static(
                self.width.div_ceil(geometry.cols_per_cube()) as u32,
                self.height.div_ceil(geometry.rows_per_cube()) as u32,
                self.batch as u32,
            ),
            CubeDim::new_2d(self.plane_size as u32, geometry.planes_per_cube as u32),
        )
    }
}

pub fn input_projection(
    row: super::coordinate::Rational,
    col: super::coordinate::Rational,
    radius: usize,
) -> Projection {
    Projection::new(
        &[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL],
        &[
            PhysicalAxisMap::of(BATCH),
            row.tap_axis(OUTPUT_H, TAP_H, radius),
            col.tap_axis(OUTPUT_W, TAP_W, radius),
            PhysicalAxisMap::of(CHANNEL),
        ],
    )
}

/// The number of bytes the gathered input would require if staged into shared memory.
pub fn stage_window_bytes(
    row: super::coordinate::Rational,
    col: super::coordinate::Rational,
    taps: usize,
    radius: usize,
    geometry: TileGeometry,
    vector_size: usize,
    elem_size: usize,
) -> usize {
    let extent_of = |axis| match axis {
        BATCH => 1,
        OUTPUT_H => geometry.rows_per_cube(),
        OUTPUT_W => geometry.cols_per_cube(),
        TAP_H | TAP_W => taps,
        CHANNEL => geometry.channels_per_cube(),
        other => panic!("stage_window_bytes: {other:?} is not an axis of the interpolation space"),
    };
    let window_vectors: usize =
        Compaction::new(&input_projection(row, col, radius), vector_size, extent_of)
            .extents()
            .iter()
            .product();
    window_vectors * vector_size * elem_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputStage, definition::InterpolateBlueprint};

    /// A bicubic resize to a `128x128` map on a 32-lane plane, four planes of two rows each and
    /// four output columns a lane. The channel count is what the caller varies: it is the axis
    /// the lanes cover first, so it decides the whole split and the cube grid with it.
    fn plan(channels: usize) -> InterpolateSpace {
        InterpolateSpace {
            batch: 2,
            height: 128,
            width: 128,
            channels,
            plane_size: 32,
            taps: 4,
            geometry: TileGeometry::from_blueprint(
                InterpolateBlueprint::new(InputStage::InPlace, 4, 2, 4),
                channels,
                32,
            ),
        }
    }

    /// The four levels as a table, leaf up: each row's tile is the row below it times the count
    /// beside it, so a tiling that stops dividing an axis where it meant to keep going shows up
    /// as a row that no longer multiplies out. The taps never divide: they are the reduction,
    /// and every tap of one output position accumulates into the same register.
    #[test]
    fn the_interpolate_routine_states_four_levels() {
        assert_eq!(
            plan(16).partitioning().table(&LABELS).to_string(),
            [
                "                        b × oh × ow × th × tw × c    b ×  oh ×  ow × th × tw ×  c",
                "",
                "  ◦                     · ×  · ×  · ×  · ×  · × ·    1 ×   2 ×   4 ×  4 ×  4 ×  4",
                "  ▪  32 units           · ×  · ×  8 ×  · ×  · × 4    1 ×   2 ×  32 ×  4 ×  4 × 16",
                "  ▤  4 planes a cube    · ×  4 ×  · ×  · ×  · × ·    1 ×   8 ×  32 ×  4 ×  4 × 16",
                "  ↻  1 steps            · ×  · ×  · ×  · ×  · × 1    1 ×   8 ×  32 ×  4 ×  4 × 16",
                "  ▣  128 cubes          2 × 16 ×  4 ×  · ×  · × ·    2 × 128 × 128 ×  4 ×  4 × 16",
                "",
                "                        └─ count ───────────────┘    └─ tile ───────────────────┘",
            ]
            .join("\n")
        );
    }

    /// A channel axis wider than one pass of the plane is what the walk above the planes is
    /// for: every lane rides the channels, and the blocks past the first are the cube's steps.
    /// The columns pay for it, so the same map takes eight times the cubes along `ow`.
    #[test]
    fn a_channel_axis_wider_than_a_plane_walks_its_blocks() {
        assert_eq!(
            plan(256).partitioning().table(&LABELS).to_string(),
            [
                "                        b × oh × ow × th × tw ×  c    b ×  oh ×  ow × th × tw ×   c",
                "",
                "  ◦                     · ×  · ×  · ×  · ×  · ×  ·    1 ×   2 ×   4 ×  4 ×  4 ×   4",
                "  ▪  32 units           · ×  · ×  · ×  · ×  · × 32    1 ×   2 ×   4 ×  4 ×  4 × 128",
                "  ▤  4 planes a cube    · ×  4 ×  · ×  · ×  · ×  ·    1 ×   8 ×   4 ×  4 ×  4 × 128",
                "  ↻  2 steps            · ×  · ×  · ×  · ×  · ×  2    1 ×   8 ×   4 ×  4 ×  4 × 256",
                "  ▣  1024 cubes         2 × 16 × 32 ×  · ×  · ×  ·    2 × 128 × 128 ×  4 ×  4 × 256",
                "",
                "                        └─ count ────────────────┘    └─ tile ────────────────────┘",
            ]
            .join("\n")
        );
    }
}
