use super::coordinate::gcd;

/// How the output is spread over cubes, over a cube's planes, and over a plane's lanes.
///
/// The lane split is what keeps a cube's reads and writes contiguous. Consecutive lanes take
/// consecutive channels of one column before stepping to the next column, so a plane covers
/// `lane_channels * channel_block` adjacent elements at a time. Lanes ride the output columns only
/// for whatever width the channel axis cannot absorb.
#[derive(Clone, Copy, Debug)]
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
    pub fn heuristic(channels: usize, lanes: usize) -> Self {
        const TARGET_COLS_PER_CUBE: usize = 32;

        // A lane's channel run is one memory line, and past four `f32` a wider line buys nothing.
        let channel_block = divisor_at_most(channels, 4);
        // Lanes cover the channel axis first, then spill onto the columns. `gcd` is the widest
        // split that both divides the plane and leaves whole channel blocks per lane.
        let lane_channels = gcd(lanes, channels / channel_block);
        let lane_cols = lanes / lane_channels;
        Self {
            planes_per_cube: 4,
            rows_per_plane: 2,
            lane_cols,
            // Keep one cube's width near a memory line regardless of how many lanes first cover
            // channels. RGB spreads all lanes across columns; wider channel groups leave fewer
            // column lanes and each should walk farther in registers.
            cols_per_lane: TARGET_COLS_PER_CUBE.div_ceil(lane_cols).max(1),
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

fn divisor_at_most(n: usize, cap: usize) -> usize {
    (1..=n.min(cap))
        .rev()
        .find(|d| n.is_multiple_of(*d))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_keeps_a_cube_near_one_memory_line_wide() {
        let rgb = TileGeometry::heuristic(3, 32);
        assert_eq!(rgb.lane_cols, 32);
        assert_eq!(rgb.cols_per_lane, 1);
        assert_eq!(rgb.cols_per_cube(), 32);

        let wide = TileGeometry::heuristic(16, 32);
        assert_eq!(wide.lane_cols, 8);
        assert_eq!(wide.cols_per_lane, 4);
        assert_eq!(wide.cols_per_cube(), 32);
    }
}
