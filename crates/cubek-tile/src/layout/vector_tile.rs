//! [`VectorTile`]: the values one vector load brings, as a tile.
//!
//! An operand says how it is stored, and the kernel picks how wide it loads; the tile a load
//! covers follows from both. A plain buffer is loaded in a run along its innermost axis. A buffer
//! stored in tiles is loaded a stored tile at a time: its finest tiles, fused per axis, until they
//! hold the load's values, so an NVFP4 buffer whose word runs along `K` and whose next pieces are
//! two words along `K` and two columns loads 32 values as 16 along `K` by 2 along `N`.

use crate::{Axis, TileMisfit};

/// The values one vector load brings, as a tile: its extents, finest first, one per axis.
///
/// ```ignore
/// // a plain row-major buffer, four wide
/// VectorTile::new(&[], N, 4)                                  // [(N, 4)]
/// // NVFP4 values: a word of 8 along K, 2 words along K, 2 columns; 32 values a load
/// VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32)           // [(K, 16), (N, 2)]
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorTile {
    extents: Vec<(Axis, usize)>,
}

impl VectorTile {
    /// The tile a load of `values` values covers over a buffer stored in `stored` tiles, finest
    /// first ([`TileSpec::stored_tiles`](crate::TileSpec::stored_tiles)): its finest tiles, fused
    /// where one axis continues, until they hold `values`. Nothing stated, a run of `values`
    /// along `innermost`.
    ///
    /// # Errors
    ///
    /// A load that would cut a stored tile, or one wider than every stored tile together.
    pub fn new(
        stored: &[(Axis, usize)],
        innermost: Axis,
        values: usize,
    ) -> Result<Self, TileMisfit> {
        if stored.is_empty() || values == 1 {
            return Ok(Self {
                extents: vec![(innermost, values)],
            });
        }
        let mut extents: Vec<(Axis, usize)> = Vec::new();
        let mut held = 1;
        for &(axis, count) in stored.iter().filter(|&&(_, count)| count > 1) {
            if held == values {
                break;
            }
            if !(values / held).is_multiple_of(count) {
                return Err(TileMisfit::Overshoots {
                    axis,
                    wanted: values / held,
                    piece: count,
                });
            }
            held *= count;
            match extents.last_mut() {
                Some((last, extent)) if *last == axis => *extent *= count,
                _ => extents.push((axis, count)),
            }
        }
        if held != values {
            return Err(TileMisfit::RunsOut {
                wanted: extents.last().map_or(innermost, |&(axis, _)| axis),
            });
        }
        Ok(Self { extents })
    }

    /// The extents, finest first.
    pub fn extents(&self) -> &[(Axis, usize)] {
        &self.extents
    }

    /// How many values the load brings.
    pub fn values(&self) -> usize {
        self.extents.iter().map(|&(_, extent)| extent).product()
    }

    /// The axis a one-axis load runs along and its width, for a `reader` that places a load along
    /// one axis.
    ///
    /// # Panics
    ///
    /// When the load spans several axes, naming `reader`.
    pub fn along_one_axis(&self, reader: &str) -> (Axis, usize) {
        match self.extents.as_slice() {
            &[one] => one,
            extents => panic!(
                "{reader} places a load along one axis, and this operand loads {extents:?}: \
                 read it with a narrower vector, or through a path that places a tile"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);

    /// A plain buffer loads a run along its innermost axis, whatever the width: the one rule every
    /// operand followed before a load could span axes.
    #[test]
    fn a_plain_buffer_loads_along_its_innermost_axis() {
        assert_eq!(VectorTile::new(&[], N, 4).unwrap().extents(), &[(N, 4)]);
        assert_eq!(VectorTile::new(&[], N, 1).unwrap().extents(), &[(N, 1)]);
    }

    /// An NVFP4 load of four words, two along `K` and two columns, is a 16 by 2 rectangle; two
    /// words are one column's 16 values, and one word its 8.
    #[test]
    fn a_stored_buffer_loads_its_finest_tiles() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 32), (N, 8)];
        assert_eq!(
            VectorTile::new(&stored, N, 32).unwrap().extents(),
            &[(K, 16), (N, 2)]
        );
        assert_eq!(
            VectorTile::new(&stored, N, 16).unwrap().extents(),
            &[(K, 16)]
        );
        assert_eq!(VectorTile::new(&stored, N, 8).unwrap().extents(), &[(K, 8)]);
        assert_eq!(VectorTile::new(&stored, N, 32).unwrap().values(), 32);
    }

    /// A load never cuts a stored tile: four values of an eight-value word, or 24 values that would
    /// take one and a half of the next tile, are refused. Neither may it outgrow what is stated.
    #[test]
    fn a_load_that_cuts_a_stored_tile_is_refused() {
        let stored = [(K, 8), (N, 2)];
        assert_eq!(
            VectorTile::new(&stored, N, 4),
            Err(TileMisfit::Overshoots {
                axis: K,
                wanted: 4,
                piece: 8
            })
        );
        assert!(VectorTile::new(&stored, N, 24).is_err());
        assert!(VectorTile::new(&stored, N, 32).is_err());
    }

    /// A stated tile of one is a dim of the buffer, not a piece of a load.
    #[test]
    fn a_stored_tile_of_one_is_skipped() {
        let stored = [(N, 4), (N, 1), (K, 2)];
        assert_eq!(
            VectorTile::new(&stored, N, 8).unwrap().extents(),
            &[(N, 4), (K, 2)]
        );
    }

    #[test]
    #[should_panic(expected = "the fragment load places a load along one axis")]
    fn a_rectangle_is_refused_by_a_one_axis_reader() {
        let tile = VectorTile::new(&[(K, 8), (N, 2)], N, 16).unwrap();
        tile.along_one_axis("the fragment load");
    }
}
