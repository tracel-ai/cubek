use super::coordinate::gcd;

/// How the output is spread over cubes, over a cube's planes, and over a plane's lanes.
///
/// The lane split is what keeps a cube's reads and writes contiguous. Consecutive lanes take
/// consecutive channels of one column before stepping to the next column, so a plane covers
/// `lane_channels * channel_block` adjacent elements at a time. Lanes ride the output columns only
/// for whatever width the channel axis cannot absorb.
///
/// Every tuning choice is the caller's. Only the lane split is derived, because
/// [`interpolate_space`](super::space::interpolate_space) requires
/// `lane_cols * lane_channels == lanes` exactly and most stated combinations would not hold it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeometry {
    /// Planes per cube, each walking `rows_per_plane` output rows.
    pub planes_per_cube: usize,
    pub rows_per_plane: usize,
    /// Lanes riding the output columns; `lane_cols * lane_channels` is the plane width.
    pub lane_cols: usize,
    pub cols_per_lane: usize,
    /// Lanes riding the channels.
    pub lane_channels: usize,
    pub channel_block: usize,
}

impl TileGeometry {
    /// Build a geometry from the stated tuning choices, solving the lane split around them.
    ///
    /// `channel_block` is the lane's channel run, `None` to solve it with the rest of the split.
    pub fn from_config(
        channels: usize,
        lanes: usize,
        planes_per_cube: usize,
        rows_per_plane: usize,
        cols_per_lane: usize,
        channel_block: Option<usize>,
    ) -> Self {
        let (lane_cols, lane_channels, channel_block) = lane_split(channels, lanes, channel_block);
        Self {
            planes_per_cube,
            rows_per_plane,
            lane_cols,
            cols_per_lane,
            lane_channels,
            channel_block,
        }
    }

    pub fn rows_per_cube(&self) -> usize {
        self.planes_per_cube * self.rows_per_plane
    }

    pub fn cols_per_cube(&self) -> usize {
        self.lane_cols * self.cols_per_lane
    }

    pub fn channels_per_cube(&self) -> usize {
        self.lane_channels * self.channel_block
    }
}

/// The one derivation left: `(lane_cols, lane_channels, channel_block)` covering the plane exactly.
///
/// A stated `block` is taken as the lane's channel run and the rest of the split is solved around
/// it. It has to divide `channels`: the cube's channel cut is sized from it, and a run the axis
/// does not cover exactly leaves the walk a partial tile the plane cover then refuses.
fn lane_split(channels: usize, lanes: usize, block: Option<usize>) -> (usize, usize, usize) {
    if let Some(block) = block {
        assert!(
            block > 0 && channels.is_multiple_of(block),
            "TileGeometry: the channel block ({block}) must divide the {channels} channels"
        );
    }

    // A plane of one lane splits nothing, so `gcd` would pin both counts to 1 and leave the
    // channel axis walked in blocks of four. Cover it in one pass instead: a narrower block
    // re-reads the same tap window and re-evaluates the same separable weights per block.
    if lanes == 1 {
        return (1, 1, block.unwrap_or_else(|| divisor_at_most(channels, 32)));
    }

    // A lane's channel run is one memory line, and past four `f32` a wider line buys nothing.
    let channel_block = block.unwrap_or_else(|| divisor_at_most(channels, 4));
    // Lanes cover the channel axis first, then spill onto the columns. `gcd` is the widest
    // split that both divides the plane and leaves whole channel blocks per lane.
    let lane_channels = gcd(lanes, channels / channel_block);
    (lanes / lane_channels, lane_channels, channel_block)
}

fn divisor_at_most(n: usize, cap: usize) -> usize {
    (1..=n.min(cap))
        .rev()
        .find(|d| n.is_multiple_of(*d))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split covers the plane exactly, which is what `interpolate_space` asserts.
    #[test]
    fn the_lane_split_covers_the_plane() {
        for lanes in [1, 8, 16, 32, 64] {
            for channels in [1, 2, 3, 4, 8, 16, 32] {
                let (lane_cols, lane_channels, _) = lane_split(channels, lanes, None);
                assert_eq!(
                    lane_cols * lane_channels,
                    lanes,
                    "c={channels} over {lanes} lanes"
                );
            }
        }
    }

    /// Lanes take the channel axis first and ride the columns for the rest.
    #[test]
    fn a_plane_spreads_over_channels_before_columns() {
        let (lane_cols, lane_channels, channel_block) = lane_split(3, 32, None);
        assert_eq!((lane_cols, lane_channels, channel_block), (32, 1, 3));

        let (lane_cols, lane_channels, channel_block) = lane_split(16, 32, None);
        assert_eq!((lane_cols, lane_channels, channel_block), (8, 4, 4));
    }

    /// A single-lane plane covers the whole channel axis in one pass.
    #[test]
    fn a_plane_of_one_lane_covers_every_channel() {
        for channels in [2, 3, 16] {
            let geometry = TileGeometry::from_config(channels, 1, 4, 8, 8, None);
            assert_eq!(geometry.lane_cols, 1);
            assert_eq!(geometry.lane_channels, 1);
            assert_eq!(geometry.channels_per_cube(), channels);
        }
    }

    /// A stated block is the lane's run, and the split still has to cover the plane exactly.
    #[test]
    fn a_stated_channel_block_solves_the_split_around_it() {
        for (channels, block, expected) in [(16, 1, 16), (16, 2, 8), (16, 8, 2), (3, 1, 1)] {
            let (lane_cols, lane_channels, channel_block) = lane_split(channels, 32, Some(block));
            assert_eq!(channel_block, block, "c={channels} block={block}");
            assert_eq!(lane_channels, expected, "c={channels} block={block}");
            assert_eq!(lane_cols * lane_channels, 32, "c={channels} block={block}");
        }
    }

    /// The cube's channel cut is sized from the block, so a run the axis cannot cover is refused
    /// here rather than surfacing as a plane cover failure.
    #[test]
    #[should_panic(expected = "must divide the 3 channels")]
    fn a_channel_block_that_does_not_divide_is_refused() {
        lane_split(3, 32, Some(4));
    }

    /// The stated choices reach the geometry untouched.
    #[test]
    fn the_config_states_every_tuning_choice() {
        let geometry = TileGeometry::from_config(3, 32, 2, 4, 8, None);
        assert_eq!(geometry.planes_per_cube, 2);
        assert_eq!(geometry.rows_per_plane, 4);
        assert_eq!(geometry.cols_per_lane, 8);
        assert_eq!(geometry.rows_per_cube(), 8);
    }
}
