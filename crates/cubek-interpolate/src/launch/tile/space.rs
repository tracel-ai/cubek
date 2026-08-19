use super::geometry::TileGeometry;
use cubek_tile::*;

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);
pub const LEAF: Leaf = Leaf::memory(MemoryMmaConfig::new(64, false, true));

pub fn interpolate_space(
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    lanes: usize,
    taps: usize,
    geometry: TileGeometry,
) -> Space {
    let (cols_per_cube, channels_per_cube) = if geometry.lanes_on_channels {
        (geometry.cols_per_lane, lanes * geometry.channel_block)
    } else {
        (lanes * geometry.cols_per_lane, geometry.channel_block)
    };
    Tiling::new()
        .extents(&[
            (BATCH, batch),
            (OUTPUT_H, height),
            (OUTPUT_W, width),
            (TAP_H, taps),
            (TAP_W, taps),
            (CHANNEL, channels),
        ])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(BATCH, Cut::cube(CubeAxis::Z, 1))
                .axis(OUTPUT_H, Cut::cube(CubeAxis::Y, geometry.rows_per_cube))
                .axis(OUTPUT_W, Cut::cube(CubeAxis::X, cols_per_cube))
                .axis(TAP_H, Cut::sequential(taps))
                .axis(TAP_W, Cut::sequential(taps))
                .axis(CHANNEL, Cut::sequential(channels_per_cube))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(BATCH, Cut::sequential(1))
                .axis(OUTPUT_H, Cut::sequential(geometry.rows_per_cube))
                .axis(
                    OUTPUT_W,
                    if geometry.lanes_on_channels {
                        Cut::sequential(cols_per_cube)
                    } else {
                        Cut::unit(geometry.cols_per_lane)
                    },
                )
                .axis(TAP_H, Cut::sequential(taps))
                .axis(TAP_W, Cut::sequential(taps))
                .axis(
                    CHANNEL,
                    if geometry.lanes_on_channels {
                        Cut::unit(geometry.channel_block)
                    } else {
                        Cut::sequential(geometry.channel_block)
                    },
                )
        })
        .build()
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
