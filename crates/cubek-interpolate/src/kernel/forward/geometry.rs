use super::coordinate::gcd;
use crate::definition::InterpolateBlueprint;

/// How the output is spread over cubes, over a cube's planes, and over a plane's units.
///
/// The unit split is what keeps a cube's reads and writes contiguous. Consecutive units take
/// consecutive channels of one column before stepping to the next column, so a plane covers
/// `unit_channels * channel_block` adjacent elements at a time. Units ride the output columns only
/// for whatever width the channel axis cannot absorb.
///
/// The blueprint states everything but the unit split, which is derived here because
/// [`interpolate_space`](super::space::interpolate_space) requires
/// `unit_cols * unit_channels == units` exactly and most stated combinations would not hold it.
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
    /// Build a geometry from a resolved blueprint, solving the unit split around it.
    ///
    /// The blueprint's channel block is the unit's channel run, `None` to solve it with the rest
    /// of the split.
    pub fn from_blueprint(blueprint: InterpolateBlueprint, channels: usize, units: usize) -> Self {
        let (unit_cols, unit_channels, channel_block) =
            unit_split(channels, units, blueprint.channel_block);
        Self {
            planes_per_cube: blueprint.planes_per_cube,
            rows_per_plane: blueprint.rows_per_plane,
            unit_cols,
            cols_per_unit: blueprint.cols_per_unit,
            unit_channels,
            channel_block,
        }
    }

    pub fn rows_per_cube(&self) -> usize {
        self.planes_per_cube * self.rows_per_plane
    }

    pub fn cols_per_cube(&self) -> usize {
        self.unit_cols * self.cols_per_unit
    }

    pub fn channels_per_cube(&self) -> usize {
        self.unit_channels * self.channel_block
    }
}

/// The one derivation left: `(unit_cols, unit_channels, channel_block)` covering the plane exactly.
///
/// A stated `block` is taken as the unit's channel run and the rest of the split is solved around
/// it. It need not divide `channels`: a run the axis does not cover exactly leaves the last block
/// part padding, which the space reports as an overhang and every read and write then masks. That
/// is the point of stating one, since a block of `4` over `3` channels is what lets the operand be
/// staged in whole lines an `NHWC` buffer could never hand out
/// ([`stage_vectorize`](cubek_tile::StridedTileSource::stage_vectorize)).
fn unit_split(channels: usize, units: usize, block: Option<usize>) -> (usize, usize, usize) {
    if let Some(block) = block {
        assert!(
            block > 0,
            "TileGeometry: the channel block must be at least one element"
        );
    }

    // A plane of one unit splits nothing, so `gcd` would pin both counts to 1 and leave the
    // channel axis walked in blocks of four. Cover it in one pass instead: a narrower block
    // re-reads the same tap window and re-evaluates the same separable weights per block.
    if units == 1 {
        return (1, 1, block.unwrap_or_else(|| divisor_at_most(channels, 32)));
    }

    // A unit's channel run is one memory line, and past four `f32` a wider line buys nothing.
    let channel_block = block.unwrap_or_else(|| divisor_at_most(channels, 4));
    // Units cover the channel axis first, then spill onto the columns. `gcd` is the widest
    // split that both divides the plane and leaves whole channel blocks per unit. The count is
    // rounded up: a block the axis does not fill exactly is still one block, its tail padding.
    let unit_channels = gcd(units, channels.div_ceil(channel_block));
    (units / unit_channels, unit_channels, channel_block)
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
    use crate::InputStage;

    /// The split covers the plane exactly, which is what `interpolate_space` asserts.
    #[test]
    fn the_unit_split_covers_the_plane() {
        for units in [1, 8, 16, 32, 64] {
            for channels in [1, 2, 3, 4, 8, 16, 32] {
                let (unit_cols, unit_channels, _) = unit_split(channels, units, None);
                assert_eq!(
                    unit_cols * unit_channels,
                    units,
                    "c={channels} over {units} units"
                );
            }
        }
    }

    /// Units take the channel axis first and ride the columns for the rest.
    #[test]
    fn a_plane_spreads_over_channels_before_columns() {
        let (unit_cols, unit_channels, channel_block) = unit_split(3, 32, None);
        assert_eq!((unit_cols, unit_channels, channel_block), (32, 1, 3));

        let (unit_cols, unit_channels, channel_block) = unit_split(16, 32, None);
        assert_eq!((unit_cols, unit_channels, channel_block), (8, 4, 4));
    }

    /// A single-unit plane covers the whole channel axis in one pass.
    #[test]
    fn a_plane_of_one_unit_covers_every_channel() {
        for channels in [2, 3, 16] {
            let blueprint = InterpolateBlueprint::new(InputStage::InPlace, 4, 8, 8);
            let geometry = TileGeometry::from_blueprint(blueprint, channels, 1);
            assert_eq!(geometry.unit_cols, 1);
            assert_eq!(geometry.unit_channels, 1);
            assert_eq!(geometry.channels_per_cube(), channels);
        }
    }

    /// A stated block is the unit's run, and the split still has to cover the plane exactly.
    #[test]
    fn a_stated_channel_block_solves_the_split_around_it() {
        for (channels, block, expected) in [(16, 1, 16), (16, 2, 8), (16, 8, 2), (3, 1, 1)] {
            let (unit_cols, unit_channels, channel_block) = unit_split(channels, 32, Some(block));
            assert_eq!(channel_block, block, "c={channels} block={block}");
            assert_eq!(unit_channels, expected, "c={channels} block={block}");
            assert_eq!(unit_cols * unit_channels, 32, "c={channels} block={block}");
        }
    }

    /// A block wider than the axis is the padded case: one unit covers the whole channel run and
    /// the rest ride the columns, exactly as they do for the unpadded block of `3`. The tail of
    /// that block is padding, which the space reports as an overhang.
    #[test]
    fn a_channel_block_may_overhang_the_axis() {
        assert_eq!(unit_split(3, 32, Some(4)), (32, 1, 4));
        assert_eq!(unit_split(3, 32, Some(3)), (32, 1, 3));
    }

    /// A block the axis covers in whole multiples splits the plane over those multiples, padded or
    /// not: `6` channels in blocks of `4` is two blocks, the second half padding.
    #[test]
    fn a_partly_covered_axis_still_splits_over_whole_blocks() {
        let (unit_cols, unit_channels, block) = unit_split(6, 32, Some(4));
        assert_eq!((unit_channels, block), (2, 4));
        assert_eq!(unit_cols * unit_channels, 32);
    }

    /// The blueprint's choices reach the geometry untouched.
    #[test]
    fn the_blueprint_states_every_tuning_choice() {
        let blueprint = InterpolateBlueprint::new(InputStage::InPlace, 2, 4, 8);
        let geometry = TileGeometry::from_blueprint(blueprint, 3, 32);
        assert_eq!(geometry.planes_per_cube, 2);
        assert_eq!(geometry.rows_per_plane, 4);
        assert_eq!(geometry.cols_per_unit, 8);
        assert_eq!(geometry.rows_per_cube(), 8);
    }
}
