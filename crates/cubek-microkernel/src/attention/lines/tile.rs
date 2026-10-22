use crate::attention::{S, StepLines, StepLinesExpand, StepWalk};
use core::marker::PhantomData;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// A float operand's lines, read where they lie through its tile, at `E`, the element it holds.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TileLines<E: Float> {
    pub tile: Tile,
    #[cube(comptime)]
    pub element: PhantomData<E>,
}

#[cube]
impl<E: Float, N: Size> StepLines<N> for TileLines<E> {
    type Element = E;

    fn load(
        &self,
        step: &Region,
        walk: &StepWalk,
        #[comptime] lines: usize,
    ) -> Array<Vector<E, N>> {
        let window = self.tile.at(step);
        let read = window.flat::<E, N>();
        let first_key = step.origin(S);
        let (teams, keys_in_flight) = comptime!((walk.teams, walk.keys_in_flight));
        let units_per_key = comptime!(walk.units_per_key);
        let per_key = comptime!(lines.div_ceil(units_per_key));
        let mut loaded = Array::<Vector<E, N>>::new(comptime!(keys_in_flight * per_key));
        #[unroll]
        for c in 0..keys_in_flight {
            let at = c * teams + walk.team;
            let seen = ((first_key + at) as u32) < walk.attended;
            #[unroll]
            for p in 0..per_key {
                let line = walk.lane + p * units_per_key;
                let mut value = Vector::<E, N>::cast_from(0u32);
                if line < lines && seen {
                    value = read.read(at * lines + line);
                }
                loaded[c * per_key + p] = value;
            }
        }
        loaded
    }

    fn at(&self, region: &Region) -> TileLines<E> {
        TileLines::<E> {
            tile: self.tile.at(region),
            element: PhantomData,
        }
    }
}
