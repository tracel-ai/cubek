use super::base::CubeQkNormedConvMicrokernel;
use super::past_taps::PastTaps;
use super::tiles::QkNormedConvTiles;
use crate::activation::silu;
use crate::linear_recurrence::{CHANNEL, HEAD_CHANNEL};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// A key head's convolved queries then keys in shared memory, and each line's sum of squares.
#[derive(CubeType)]
pub(super) struct ConvolvedHead<N: Size> {
    queries_then_keys: Shared<[Vector<f32, N>]>,
    squares: Shared<[f32]>,
    #[cube(comptime)]
    microkernel: CubeQkNormedConvMicrokernel,
}

#[cube]
impl<N: Size> ConvolvedHead<N> {
    pub fn new(#[comptime] microkernel: CubeQkNormedConvMicrokernel) -> Self {
        let width = N::value();
        let lines = comptime!(2 * microkernel.heads.key_features / width);
        ConvolvedHead::<N> {
            queries_then_keys: Shared::<[Vector<f32, N>]>::new_slice(lines),
            squares: Shared::<[f32]>::new_slice(lines),
            microkernel,
        }
    }

    pub fn query_line(&self, part: usize) -> Vector<f32, N> {
        self.queries_then_keys[part]
    }

    pub fn key_line(&self, part: usize) -> Vector<f32, N> {
        let width = N::value();
        let lines = comptime!(self.microkernel.heads.key_features / width);
        self.queries_then_keys[lines + part]
    }

    pub fn query_squares(&self, part: usize) -> f32 {
        self.squares[part]
    }

    pub fn key_squares(&self, part: usize) -> f32 {
        let width = N::value();
        let lines = comptime!(self.microkernel.heads.key_features / width);
        self.squares[lines + part]
    }

    /// Each unit's lines of the head's channels: the causal taps over their past and the newest
    /// value, then the silu. Over a window the newest value is written as the ring's next column
    /// after the taps are read, by the unit that owns the channel.
    pub fn convolve<E: Float>(
        &mut self,
        cube: &Region,
        tiles: &mut QkNormedConvTiles,
        head: usize,
    ) {
        let microkernel = comptime!(self.microkernel);
        let width = N::value();
        let read_columns = microkernel.taps - 1;
        let past = PastTaps::new(cube, &*tiles, read_columns);
        let mut values = tiles.value.flat_mut::<f32, N>();
        for share in cube {
            let part = share.coord(HEAD_CHANNEL);
            let first = self.first_channel(part * width, head);
            let line = first / width;
            let newest = past.newest::<E, N>(line);
            let mut sums =
                Self::widened(newest) * Self::weights::<E>(&tiles.filters, first, read_columns);
            #[unroll]
            for column in 0..read_columns {
                let behind = Self::widened(past.behind::<E, N>(column, line));
                sums += behind * Self::weights::<E>(&tiles.filters, first, column);
            }
            let convolved = Self::activated::<E>(sums);
            past.advance::<E, N>(line, newest);
            if self.holds_query_or_key(part) {
                self.keep(part, convolved);
            } else {
                values.write(self.value_line(part), convolved);
            }
        }
    }

    fn holds_query_or_key(&self, part: usize) -> bool {
        let width = N::value();
        part * width < comptime!(2 * self.microkernel.heads.key_features)
    }

    fn value_line(&self, part: usize) -> usize {
        let width = N::value();
        part - comptime!(2 * self.microkernel.heads.key_features / width)
    }

    fn keep(&mut self, part: usize, convolved: Vector<f32, N>) {
        let width = N::value();
        self.queries_then_keys[part] = convolved;
        let mut squares = 0.0f32;
        #[unroll]
        for lane in 0..width {
            let x = convolved.extract(lane);
            squares += x * x;
        }
        self.squares[part] = squares;
    }

    fn weights<E: Float>(filters: &Tile, first: usize, tap: usize) -> Vector<f32, N> {
        let width = N::value();
        let mut weights = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for lane in 0..width {
            let row = filters.within(CHANNEL, first + lane, first + lane + 1);
            let weight = row.flat::<E, Const<1>>().read(tap).extract(0usize);
            weights.insert(lane, f32::cast_from(weight));
        }
        weights
    }

    fn widened<E: Float>(line: Vector<E, N>) -> Vector<f32, N> {
        let width = N::value();
        let mut widened = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for lane in 0..width {
            widened.insert(lane, f32::cast_from(line.extract(lane)));
        }
        widened
    }

    /// The stream channel of the head's channel `local`: the queries, keys and values come from
    /// three ranges of the stream that no one window of its tile spans.
    fn first_channel(&self, local: usize, head: usize) -> usize {
        let heads = comptime!(self.microkernel.heads);
        let (key_features, heads_per_group, value_features) = comptime!((
            heads.key_features,
            heads.heads_per_group,
            heads.value_features
        ));
        let key_dim = heads.key_dim();
        if local < key_features {
            head * key_features + local
        } else if local < 2 * key_features {
            key_dim + head * key_features + (local - key_features)
        } else {
            2 * key_dim + head * heads_per_group * value_features + (local - 2 * key_features)
        }
    }

    /// The silu of `sums`, rounded to `E` before and after as composed ops hold them.
    fn activated<E: Float>(sums: Vector<f32, N>) -> Vector<f32, N> {
        let width = N::value();
        let mut convolved = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for lane in 0..width {
            let mixed = f32::cast_from(E::cast_from(sums.extract(lane)));
            convolved.insert(lane, f32::cast_from(E::cast_from(silu(mixed))));
        }
        convolved
    }
}
