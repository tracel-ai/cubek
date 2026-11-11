use super::base::CubeQkNormedConvMicrokernel;
use super::head_channels::HeadChannels;
use super::tiles::QkNormedConvTiles;
use crate::linear_recurrence::HEAD_CHANNEL;
use crate::linear_recurrence::conv::cube::{CausalFilters, PastTaps};
use cubecl::tile::Region;
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
        channels: &HeadChannels,
    ) {
        let width = N::value();
        let microkernel = comptime!(self.microkernel);
        let read_columns = microkernel.taps - 1;
        let past = PastTaps::new(cube, &tiles.past, microkernel.past, read_columns);
        let filters = CausalFilters::new(&tiles.filters, microkernel.taps);
        let mut values = tiles.value.flat_mut::<f32, N>();
        for share in cube {
            let part = share.coord(HEAD_CHANNEL);
            let first = channels.first(part * width);
            let line = first / width;
            let newest = past.newest::<E, N>(line);
            let convolved = filters.convolved_line::<E, N>(&past, newest, line);
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
}
