use super::geometry::TileGeometry;
use cubecl::{Runtime, client::ComputeClient};
use cubek_tile::*;

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);

/// The instruction leaf, which the device decides.
///
/// A GPU keeps one nest body and the (line, lane) fan-out walk: the edge split clones the whole
/// body, which doubles the shader and makes the two halves diverge within a plane. A CPU takes the
/// opposite of each, plus the wider register block its register file has room for.
pub fn leaf<R: Runtime>(client: &ComputeClient<R>) -> Leaf {
    match client.properties().hardware.num_cpu_cores {
        Some(_) => Leaf::memory(MemoryMmaConfig::new(256, true, false)),
        None => Leaf::memory(MemoryMmaConfig::new(64, false, true)),
    }
}

/// Every level runs a single-slot ring. CHANNEL is the cube walk's only moving axis, and it moves
/// only past `lanes * 4` channels; below that the walk is one region, where a deeper pipeline has
/// no fill to hide and pays the ring's prologue and drain for nothing.
pub fn interpolate_space(
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    lanes: usize,
    taps: usize,
    geometry: TileGeometry,
) -> Space {
    assert!(
        geometry.lane_cols * geometry.lane_channels == lanes,
        "interpolate_space: the lane split covers {} of the plane's {lanes} lanes",
        geometry.lane_cols * geometry.lane_channels
    );
    let channels_per_cube = geometry.channels_per_cube();
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
                .axis(OUTPUT_H, Cut::cube(CubeAxis::Y, geometry.rows_per_cube()))
                .axis(OUTPUT_W, Cut::cube(CubeAxis::X, geometry.cols_per_cube()))
                .axis(TAP_H, Cut::sequential(taps))
                .axis(TAP_W, Cut::sequential(taps))
                .axis(CHANNEL, Cut::sequential(channels_per_cube))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(BATCH, Cut::sequential(1))
                .axis(OUTPUT_H, Cut::plane(geometry.rows_per_plane))
                .axis(OUTPUT_W, Cut::sequential(geometry.cols_per_cube()))
                .axis(TAP_H, Cut::sequential(taps))
                .axis(TAP_W, Cut::sequential(taps))
                .axis(CHANNEL, Cut::sequential(channels_per_cube))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(BATCH, Cut::sequential(1))
                .axis(OUTPUT_H, Cut::sequential(geometry.rows_per_plane))
                .axis(
                    OUTPUT_W,
                    lanes_over(geometry.lane_cols, geometry.cols_per_lane),
                )
                .axis(TAP_H, Cut::sequential(taps))
                .axis(TAP_W, Cut::sequential(taps))
                .axis(
                    CHANNEL,
                    lanes_over(geometry.lane_channels, geometry.channel_block),
                )
        })
        .build()
}

/// `edge`-sized tiles dealt to `instances` lanes of the plane. [`Cut::unit`] claims the whole plane
/// for one axis; the interpolation splits it across two, so the count is stated outright. One lane
/// is no split at all, and stays sequential so the walk keeps its comptime coordinate.
fn lanes_over(instances: usize, edge: usize) -> Cut {
    match instances {
        1 => Cut::sequential(edge),
        n => Cut::new(
            edge,
            Distribution::Spatial {
                scope: ComputeScope::Unit,
                spread: Spread::Contiguous,
                coverage: Coverage::Instances(n),
            },
        ),
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

/// Whether the gathered input is worth staging, from the tap window the cube would stage.
///
/// Staging turns `taps²` masked global reads per output cell into `window / outputs` cooperative
/// coalesced reads plus `taps²` unmasked shared-memory ones. It pays exactly when that reuse
/// factor is high: the taps of neighbouring outputs overlap, so the window holds each element
/// once. When the stride outruns the taps the window is mostly rows the fill reads and the taps
/// skip, and the cache serves an in-place read better than the fill does
/// (see [`Compaction`]'s own account of the two regimes).
///
/// The threshold is the middle of the flat optimum swept over this crate's benchmark set: `4`
/// through `6` all cost the same 4.3 ms of regret against picking each of the sixteen problems
/// perfectly, which is 2.2% over that oracle, against 7.1% for always staging and 112% for never
/// staging. Reuse does not separate the set exactly, and no threshold does: the five problems it
/// calls wrong interleave with ones it calls right (`0.93` wants a stage, `1.00` does not; `3.22`
/// wants one, `3.82` does not). Four of those five cost under 1.3 ms and sit in the 3-5 ms band
/// that reverses between runs; the residual signal is something other than tap overlap. Every
/// problem where the choice costs more than 1.7x is called right.
pub fn stage_input(
    row: super::coordinate::Rational,
    col: super::coordinate::Rational,
    taps: usize,
    radius: usize,
    geometry: TileGeometry,
    vector_size: usize,
) -> Residence {
    const STAGE_REUSE: usize = 5;

    let extent_of = |axis| match axis {
        BATCH => 1,
        OUTPUT_H => geometry.rows_per_cube(),
        OUTPUT_W => geometry.cols_per_cube(),
        TAP_H | TAP_W => taps,
        CHANNEL => geometry.channels_per_cube(),
        other => panic!("stage_input: {other:?} is not an axis of the interpolation space"),
    };
    let window: usize = Compaction::of(&input_projection(row, col, radius), vector_size, extent_of)
        .extents()
        .iter()
        .product();
    let reads = extent_of(OUTPUT_H) * extent_of(OUTPUT_W) * extent_of(CHANNEL) * taps * taps;

    match reads >= STAGE_REUSE * window {
        true => Residence::Smem,
        false => Residence::InPlace,
    }
}

#[cfg(test)]
mod tests {
    use super::super::coordinate::Rational;
    use super::*;
    use crate::definition::{InterpolateMode, InterpolateOptions, NearestMode, get_transform};

    const LANES: usize = 32;

    /// The residence [`stage_input`] derives for one benchmarked problem.
    fn pick(
        (in_h, in_w): (usize, usize),
        (out_h, out_w): (usize, usize),
        channels: usize,
        mode: InterpolateMode,
        taps: usize,
    ) -> Residence {
        let options = InterpolateOptions::new(mode);
        let radius = (taps - 1) / 2;
        stage_input(
            Rational::of(get_transform(in_h, out_h, options)),
            Rational::of(get_transform(in_w, out_w, options)),
            taps,
            radius,
            TileGeometry::heuristic(channels, LANES),
            1,
        )
    }

    /// The regimes the benchmark set separates cleanly, which are the ones where the choice is
    /// worth making: the upsamples overlap their taps heavily and want the stage, the coarse
    /// downsamples read a window the taps mostly skip and want the cache. Halving at three
    /// channels is deliberately absent for bilinear and bicubic: the derivation reads those as
    /// in-place, and the runs disagree with each other about whether that is right.
    #[test]
    fn stage_input_picks_the_separable_regimes() {
        let up_small = ((2048, 2048), (4096, 4096), 3);
        let up_wide = ((512, 512), (1024, 1024), 16);
        let down_small = ((2048, 2048), (1024, 1024), 3);
        let down_coarse = ((2048, 1024), (512, 512), 2);

        for (input, output, channels) in [up_small, up_wide] {
            for (mode, taps) in [
                (InterpolateMode::Bilinear, 2),
                (InterpolateMode::Bicubic, 4),
                (InterpolateMode::Lanczos3, 6),
            ] {
                assert_eq!(
                    pick(input, output, channels, mode, taps),
                    Residence::Smem,
                    "{mode:?} upsample to {output:?} at c={channels}"
                );
            }
        }

        // One tap is one read per output: there is no reuse for a stage to exploit.
        assert_eq!(
            pick(
                up_small.0,
                up_small.1,
                up_small.2,
                InterpolateMode::Nearest(NearestMode::Floor),
                1
            ),
            Residence::InPlace
        );

        // Halving keeps enough overlap for six taps, but not for two or four.
        assert_eq!(
            pick(
                down_small.0,
                down_small.1,
                down_small.2,
                InterpolateMode::Lanczos3,
                6
            ),
            Residence::Smem
        );
        for (mode, taps) in [
            (InterpolateMode::Bilinear, 2),
            (InterpolateMode::Bicubic, 4),
        ] {
            assert_eq!(
                pick(down_small.0, down_small.1, down_small.2, mode, taps),
                Residence::InPlace,
                "{mode:?} downsample by two at c=3"
            );
        }

        // Quartering along the rows leaves the window mostly skipped rows, six taps included.
        for (mode, taps) in [
            (InterpolateMode::Bicubic, 4),
            (InterpolateMode::Lanczos3, 6),
        ] {
            assert_eq!(
                pick(down_coarse.0, down_coarse.1, down_coarse.2, mode, taps),
                Residence::InPlace,
                "{mode:?} downsample by four at c=2"
            );
        }
    }
}
