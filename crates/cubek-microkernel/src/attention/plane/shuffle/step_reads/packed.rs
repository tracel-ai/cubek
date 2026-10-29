use crate::attention::S;
use crate::attention::plane::{StepReads, StepReadsExpand, StepWalk};
use cubecl::tile::layout::PageEntry;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The keys or the values of a packed cache, read as lines, decoding as they are read
/// ([`Tile::decoded`]): each operand's codebook indices, under its norms.
///
/// With the QJL fix-up, the keys' lines run twice the head dim: the first half the keys' tile,
/// the second the fix-up's, the same words read through the sign codebook under each key's
/// residual length. A query whose second half is its projection then scores both terms in one
/// dot.
///
/// A step's window holds one batch and one head, so a key is a row of it, read a line at a time:
/// the row's address and norm are computed once for all its lines. A line is read only where it
/// lies in the head dim and its key is attended, so the decoded read skips the bounds test it
/// would repeat.
#[derive(CubeType, Clone)]
pub struct PackedStepReads {
    pub tile: Tile,
    pub fixup: ComptimeOption<Tile>,
}

#[cube]
impl<N: Size> StepReads<N> for PackedStepReads {
    type Element = f32;

    fn load(
        &self,
        step: &Region,
        walk: &StepWalk,
        #[comptime] lines: usize,
    ) -> Array<Vector<f32, N>> {
        let qjl = self.fixup.is_some();
        let window = self.tile.at(step);
        // Without a fix-up, the operand stands in for it, and is never read through it.
        let fixup = if comptime!(qjl) {
            self.fixup.clone().unwrap().at(step)
        } else {
            window.clone()
        };
        let (read, fixup) = (window.decoded::<f32, N>(), fixup.decoded::<f32, N>());
        let half = comptime!(if qjl { lines / 2 } else { lines });
        let first_key = step.origin(S);
        let (teams, keys_in_flight) = comptime!((walk.teams, walk.keys_in_flight));
        let units_per_key = comptime!(walk.units_per_key);
        let per_key = lines.div_ceil(units_per_key);
        // Where the units' lines tile the row exactly, every line a unit takes lies in it: only
        // the key is tested, once, and a key's lines are read in one block, sharing its row.
        let whole_rows = lines.is_multiple_of(units_per_key);

        let mut loaded = Array::<Vector<f32, N>>::new(keys_in_flight * per_key);
        #[unroll]
        for c in 0..keys_in_flight {
            let at = c * teams + walk.team;
            #[unroll]
            for p in 0..per_key {
                loaded[c * per_key + p] = Vector::<f32, N>::cast_from(0u32);
            }
            if ((first_key + at) as u32) < walk.attended {
                #[unroll]
                for p in 0..per_key {
                    let line = walk.lane + p * units_per_key;
                    if whole_rows || line < lines {
                        let value = if comptime!(qjl) {
                            if line < half {
                                read.read_unchecked(at, line)
                            } else {
                                fixup.read_unchecked(at, line - half)
                            }
                        } else {
                            read.read_unchecked(at, line)
                        };
                        loaded[c * per_key + p] = value;
                    }
                }
            }
        }

        loaded
    }

    fn read_page_entry(&self, _region: &Region) -> PageEntry {
        PageEntry::unlisted()
    }

    fn with_page_entry(&self, _entry: &PageEntry) -> PackedStepReads {
        PackedStepReads {
            tile: self.tile.clone(),
            fixup: self.fixup.clone(),
        }
    }

    fn at(&self, region: &Region) -> PackedStepReads {
        PackedStepReads {
            tile: self.tile.at(region),
            fixup: self.fixup.clone().map(|fixup| fixup.at(region)),
        }
    }
}
