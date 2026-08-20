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
    pub fn from_config(
        channels: usize,
        lanes: usize,
        planes_per_cube: usize,
        rows_per_plane: usize,
        cols_per_lane: usize,
    ) -> Self {
        let (lane_cols, lane_channels, channel_block) = lane_split(channels, lanes);
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
fn lane_split(channels: usize, lanes: usize) -> (usize, usize, usize) {
    // A plane of one lane splits nothing, so `gcd` would pin both counts to 1 and leave the
    // channel axis walked in blocks of four. Cover it in one pass instead: a narrower block
    // re-reads the same tap window and re-evaluates the same separable weights per block.
    if lanes == 1 {
        return (1, 1, divisor_at_most(channels, 32));
    }

    // A lane's channel run is one memory line, and past four `f32` a wider line buys nothing.
    let channel_block = divisor_at_most(channels, 4);
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
                let (lane_cols, lane_channels, _) = lane_split(channels, lanes);
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
        let (lane_cols, lane_channels, channel_block) = lane_split(3, 32);
        assert_eq!((lane_cols, lane_channels, channel_block), (32, 1, 3));

        let (lane_cols, lane_channels, channel_block) = lane_split(16, 32);
        assert_eq!((lane_cols, lane_channels, channel_block), (8, 4, 4));
    }

    /// A single-lane plane covers the whole channel axis in one pass.
    #[test]
    fn a_plane_of_one_lane_covers_every_channel() {
        for channels in [2, 3, 16] {
            let geometry = TileGeometry::from_config(channels, 1, 4, 8, 8);
            assert_eq!(geometry.lane_cols, 1);
            assert_eq!(geometry.lane_channels, 1);
            assert_eq!(geometry.channels_per_cube(), channels);
        }
    }

    /// The stated choices reach the geometry untouched.
    #[test]
    fn the_config_states_every_tuning_choice() {
        let geometry = TileGeometry::from_config(3, 32, 2, 4, 8);
        assert_eq!(geometry.planes_per_cube, 2);
        assert_eq!(geometry.rows_per_plane, 4);
        assert_eq!(geometry.cols_per_lane, 8);
        assert_eq!(geometry.rows_per_cube(), 8);
    }
}
