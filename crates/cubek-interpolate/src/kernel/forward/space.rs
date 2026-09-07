use super::geometry::TileGeometry;
use cubecl::client::Client;
use cubecl::{CubeCount, CubeDim};
use cubek_tile::{Axis, Compaction, Cut, Level, PhysicalAxisMap, Projection, RegisterBlock, Space};

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

    /// Four levels, outermost first: the cube grid, the channel blocks a cube walks (one region
    /// below `lanes * 4` channels), this plane's rows, then this lane's columns and channel
    /// lines.
    pub fn levels(&self) -> Vec<Level> {
        vec![
            self.cubes(),
            self.channel_blocks(),
            self.planes(),
            self.lanes(),
        ]
    }

    pub fn space(&self) -> Space {
        Space::new(&self.extents())
    }

    /// The tile every operand is cut to at the bottom.
    pub fn leaf(&self) -> Vec<(Axis, usize)> {
        self.space().leaf(&self.levels()).extents()
    }

    /// The axes some tile reaches past the end of.
    pub fn overhangs(&self) -> Vec<Axis> {
        let (space, levels) = (self.space(), self.levels());
        space
            .axes()
            .filter(|&axis| space.overhangs(&levels, axis))
            .collect()
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

    /// This cube's box of the output, the taps whole.
    pub fn cubes(&self) -> Level {
        let geometry = self.geometry;
        Level::cubes(&[
            (OUTPUT_W, geometry.cols_per_cube()),
            (OUTPUT_H, geometry.rows_per_cube()),
        ])
        .batches(&[BATCH])
    }

    /// The cube's box walked one channel block at a time.
    pub fn channel_blocks(&self) -> Level {
        Level::walk(&[(CHANNEL, self.geometry.channels_per_cube())])
    }

    /// The cube's rows across its planes.
    pub fn planes(&self) -> Level {
        let geometry = self.geometry;
        Level::planes(&[
            Cut::new(OUTPUT_H, geometry.rows_per_plane).across(geometry.planes_per_cube)
        ])
    }

    /// This lane's columns and channel lines. The interpolation splits the plane across two
    /// axes, so the counts are stated outright; one lane along an axis is no split at all, and
    /// the axis is handed down whole.
    pub fn lanes(&self) -> Level {
        let (geometry, plane_size) = (self.geometry, self.plane_size);
        assert!(
            geometry.lane_cols * geometry.lane_channels == plane_size,
            "InterpolateSpace: the lane split covers {} of the plane's {plane_size} lanes",
            geometry.lane_cols * geometry.lane_channels
        );
        let cuts: Vec<Cut> = [
            (OUTPUT_W, geometry.lane_cols, geometry.cols_per_lane),
            (CHANNEL, geometry.lane_channels, geometry.channel_block),
        ]
        .into_iter()
        .filter(|&(_, instances, _)| instances > 1)
        .map(|(axis, instances, edge)| Cut::new(axis, edge).across(instances))
        .collect();
        Level::lanes(&cuts)
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
        Compaction::of(&input_projection(row, col, radius), vector_size, extent_of)
            .extents()
            .iter()
            .product();
    window_vectors * vector_size * elem_size
}
