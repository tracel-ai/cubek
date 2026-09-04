use super::geometry::TileGeometry;
use cubecl::client::Client;
use cubek_tile::{
    Axis, Compaction, CubeAxis, Level, LevelCuts, PhysicalAxisMap, Projection, RegisterBlock,
    Space, cubes, lanes, planes,
};

const AXES: [Axis; 6] = [BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL];

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);

/// The register block the leaf runs under, which the device decides.
pub fn register_block(client: &Client) -> RegisterBlock {
    match client.properties().hardware.num_cpu_cores {
        Some(_) => RegisterBlock::new(256).split_edge(),
        None => RegisterBlock::new(64).lane_fanout(),
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

    /// Three levels, outermost first. CHANNEL is the cube walk's only moving axis, and it moves
    /// only past `lanes * 4` channels; below that the walk is one region.
    pub fn levels(&self) -> Vec<Level> {
        vec![self.cube_level(), self.plane_level(), self.lane_level()]
    }

    pub fn space(&self) -> Space {
        Space::new(&self.extents())
    }

    /// This cube's box of the output, walked over the taps and its channel blocks.
    pub fn cube_level(&self) -> Level {
        let (taps, geometry) = (self.taps, self.geometry);
        Level::cuts(&AXES, |level| {
            level
                .distribute(cubes(CubeAxis::Z), &[(BATCH, 1)])
                .distribute(cubes(CubeAxis::Y), &[(OUTPUT_H, geometry.rows_per_cube())])
                .distribute(cubes(CubeAxis::X), &[(OUTPUT_W, geometry.cols_per_cube())])
                .walk(&[
                    (TAP_H, taps),
                    (TAP_W, taps),
                    (CHANNEL, geometry.channels_per_cube()),
                ]);
        })
    }

    /// This plane's rows.
    pub fn plane_level(&self) -> Level {
        let (taps, geometry) = (self.taps, self.geometry);
        Level::cuts(&AXES, |level| {
            level
                .distribute(planes(), &[(OUTPUT_H, geometry.rows_per_plane)])
                .walk(&[
                    (BATCH, 1),
                    (OUTPUT_W, geometry.cols_per_cube()),
                    (TAP_H, taps),
                    (TAP_W, taps),
                    (CHANNEL, geometry.channels_per_cube()),
                ]);
        })
    }

    /// This lane's columns and channel lines.
    pub fn lane_level(&self) -> Level {
        let (taps, geometry, plane_size) = (self.taps, self.geometry, self.plane_size);
        assert!(
            geometry.lane_cols * geometry.lane_channels == plane_size,
            "InterpolateSpace: the lane split covers {} of the plane's {plane_size} lanes",
            geometry.lane_cols * geometry.lane_channels
        );
        Level::cuts(&AXES, |level| {
            lanes_over(level, OUTPUT_W, geometry.lane_cols, geometry.cols_per_lane);
            lanes_over(
                level,
                CHANNEL,
                geometry.lane_channels,
                geometry.channel_block,
            );
            level.walk(&[
                (BATCH, 1),
                (OUTPUT_H, geometry.rows_per_plane),
                (TAP_H, taps),
                (TAP_W, taps),
            ]);
        })
    }
}

/// `edge`-sized tiles of `axis` dealt to `instances` lanes of the plane. The interpolation
/// splits the plane across two axes, so the count is stated outright. One lane is no split at
/// all, and is walked so the coordinate stays comptime.
fn lanes_over(level: &mut LevelCuts, axis: Axis, instances: usize, edge: usize) {
    match instances {
        1 => level.walk(&[(axis, edge)]),
        n => level.distribute(lanes(n), &[(axis, edge)]),
    };
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
        Compaction::of(&input_projection(row, col, radius), vector_size, extent_of)
            .extents()
            .iter()
            .product();
    window_vectors * vector_size * elem_size
}
