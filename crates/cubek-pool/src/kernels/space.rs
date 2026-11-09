use super::geometry::TileGeometry;
use cubecl::tile::{Axis, Level, Levels, Partitioning, Projection, Space, layout::PhysicalAxisMap};
use cubek_microkernel::pool::PoolWorkItem;

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);

/// How a window of one pooled axis lands on the input: output position `o` and tap `t` read input
/// position `o · stride + t · dilation - padding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowAxis {
    pub taps: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl WindowAxis {
    /// The input axis this window reads, as the gather maps it from `output` and `tap`.
    pub fn input_axis(&self, output: Axis, tap: Axis) -> PhysicalAxisMap {
        PhysicalAxisMap::affine(&[(output, self.stride), (tap, self.dilation)])
            .shifted(-(self.padding as isize))
    }

    /// Whether every tap of `outputs` positions reads inside an input `input` long.
    pub fn in_bounds(&self, outputs: usize, input: usize) -> bool {
        let last = outputs.saturating_sub(1) * self.stride + (self.taps - 1) * self.dilation;
        self.padding == 0 && last < input
    }
}

/// The space one launch runs over, in the terms its partitioning is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolSpace {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub plane_size: usize,
    pub taps: [usize; 2],
    pub geometry: TileGeometry,
}

impl PoolSpace {
    /// The space's axes and their extents, every one static.
    pub fn extents(&self) -> Vec<(Axis, usize)> {
        vec![
            (BATCH, self.batch),
            (OUTPUT_H, self.height),
            (OUTPUT_W, self.width),
            (TAP_H, self.taps[0]),
            (TAP_W, self.taps[1]),
            (CHANNEL, self.channels),
        ]
    }

    /// From the leaf up, in counts: one unit's columns by its channel run by a plane's rows, the
    /// units of a plane across columns and channels, the planes of a cube across rows, every
    /// channel block of the cube's box, and a cube per box. The taps never divide: every tap of
    /// one output position reduces into the same cell.
    pub fn levels(&self) -> Vec<Level> {
        let (geometry, plane_size) = (self.geometry, self.plane_size);
        assert!(
            geometry.unit_cols * geometry.unit_channels == plane_size,
            "PoolSpace: the unit split covers {} of the plane's {plane_size} units",
            geometry.unit_cols * geometry.unit_channels
        );
        let plane_units: Vec<(Axis, usize)> = [
            (OUTPUT_W, geometry.unit_cols),
            (CHANNEL, geometry.unit_channels),
        ]
        .into_iter()
        .filter(|&(_, plane_units)| plane_units > 1)
        .collect();
        Levels::leaf(&[
            (OUTPUT_W, geometry.cols_per_unit),
            (CHANNEL, geometry.channel_block),
            (OUTPUT_H, geometry.rows_per_plane),
        ])
        .units(&plane_units)
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
}

/// What one unit of `partitioning` is handed: its leaf's output positions, each its channel run.
pub fn work_item(partitioning: &Partitioning) -> PoolWorkItem {
    let leaf = partitioning.leaf();
    let channels = leaf.extent(CHANNEL);
    PoolWorkItem {
        channels,
        outputs: leaf.extent(OUTPUT_H) * leaf.extent(OUTPUT_W) * channels,
    }
}

/// The NHWC input as every output position's window reads it.
pub fn input_projection(rows: WindowAxis, cols: WindowAxis) -> Projection {
    Projection::new(
        &[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL],
        &[
            PhysicalAxisMap::of(BATCH),
            rows.input_axis(OUTPUT_H, TAP_H),
            cols.input_axis(OUTPUT_W, TAP_W),
            PhysicalAxisMap::of(CHANNEL),
        ],
    )
}
