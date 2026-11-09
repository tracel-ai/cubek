use crate::kernels::CubeMaxPoolRoutineSettings;
use cubek_std::selector::divisor_at_most;

/// How the output is spread over cubes, over a cube's planes, and over a plane's units.
///
/// Consecutive units take consecutive channels of one column before stepping to the next column,
/// so a plane covers `unit_channels * channel_block` adjacent elements at a time. Units ride the
/// output columns only for whatever width the channel axis cannot absorb.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileGeometry {
    /// Planes per cube, each walking `rows_per_plane` output rows.
    pub planes_per_cube: usize,
    pub rows_per_plane: usize,
    /// Units riding the output columns; `unit_cols * unit_channels` is the plane width.
    pub unit_cols: usize,
    pub cols_per_unit: usize,
    /// Units riding the channels.
    pub unit_channels: usize,
    pub channel_block: usize,
}

impl TileGeometry {
    /// The geometry `settings` state over `channels` on planes of `units`, the unit split solved
    /// around them.
    pub fn new(settings: &CubeMaxPoolRoutineSettings, channels: usize, units: usize) -> Self {
        let (unit_cols, unit_channels, channel_block) = unit_split(channels, units);
        Self {
            planes_per_cube: settings.planes,
            rows_per_plane: settings.rows_per_plane,
            unit_cols,
            cols_per_unit: settings.cols_per_unit,
            unit_channels,
            channel_block,
        }
    }
}

/// `(unit_cols, unit_channels, channel_block)` covering the plane exactly: units cover the channel
/// axis first, a line of at most four channels each, and spill onto the columns.
fn unit_split(channels: usize, units: usize) -> (usize, usize, usize) {
    if units == 1 {
        return (1, 1, divisor_at_most(channels, 32));
    }
    let channel_block = divisor_at_most(channels, 4);
    let unit_channels = gcd(units, channels / channel_block);
    (units / unit_channels, unit_channels, channel_block)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split covers the plane exactly, which the space asserts.
    #[test]
    fn the_unit_split_covers_the_plane() {
        for units in [1, 8, 16, 32, 64] {
            for channels in [1, 2, 3, 4, 8, 16, 32, 96] {
                let (unit_cols, unit_channels, block) = unit_split(channels, units);
                assert_eq!(
                    unit_cols * unit_channels,
                    units,
                    "c={channels} over {units}"
                );
                assert_eq!(channels % block, 0, "c={channels} over {units}");
            }
        }
    }
}
