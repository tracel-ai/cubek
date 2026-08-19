/// How the output is spread over cubes and over a plane's lanes.
#[derive(Clone, Copy, Debug)]
pub struct TileGeometry {
    pub rows_per_cube: usize,
    pub cols_per_lane: usize,
    pub channel_block: usize,
    pub lanes_on_channels: bool,
}

impl TileGeometry {
    pub fn heuristic(channels: usize, lanes: usize) -> Self {
        if channels >= lanes && channels.is_multiple_of(lanes) {
            Self {
                rows_per_cube: 4,
                cols_per_lane: 1,
                channel_block: divisor_at_most(channels / lanes, 4),
                lanes_on_channels: true,
            }
        } else {
            Self {
                rows_per_cube: 2,
                cols_per_lane: 1,
                channel_block: divisor_at_most(channels, 16),
                lanes_on_channels: false,
            }
        }
    }
}

fn divisor_at_most(n: usize, cap: usize) -> usize {
    (1..=n.min(cap))
        .rev()
        .find(|d| n.is_multiple_of(*d))
        .unwrap_or(1)
}
