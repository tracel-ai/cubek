use super::partials::Partials;
use super::row_pass::{RowRead, RowUpdate};
use super::shape::HeldStateShape;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// One unit's key rows of its run's state, a line of the run's columns a part of a row, in
/// registers for the whole walk. Every tile it reads or writes is windowed to the unit, so rows
/// the head leaves past its last read as zeros and are never written.
#[derive(CubeType)]
pub struct HeldState<N: Size> {
    lines: Array<Vector<f32, N>>,
    #[cube(comptime)]
    shape: HeldStateShape,
}

#[cube]
impl<N: Size> HeldState<N> {
    /// The unit's rows of `state`, windowed to the unit: the state read once, or zeros where the
    /// row is `fresh`, which reads nothing.
    pub fn load(state: &Tile, fresh: bool, #[comptime] shape: HeldStateShape) -> Self {
        let width = N::value();
        let (rows, columns) = comptime!((shape.rows(), shape.columns()));
        let parts = comptime!(columns / width);

        let reads = state.flat::<f32, N>();
        let zero = Vector::<f32, N>::cast_from(0.0f32);
        let mut lines = Array::<Vector<f32, N>>::new(rows * parts);

        if fresh {
            #[unroll]
            for line in 0..(rows * parts) {
                lines[line] = zero;
            }
        } else {
            #[unroll]
            for line in 0..(rows * parts) {
                lines[line] = reads.read(line);
            }
        }

        HeldState::<N> { lines, shape }
    }

    /// One pass over the unit's rows: each cell updated as `update` says, then read against the
    /// position's `read` window, the part of `Σ S[r] · read[r]` over the unit's rows left in
    /// `partials`. `keys` and `queries` are windowed to the unit and the position; `factor` and
    /// `write` are the ones `update` names.
    #[allow(clippy::too_many_arguments)]
    pub fn pass(
        &mut self,
        keys: &Tile,
        queries: &Tile,
        factor: f32,
        write: &Array<Vector<f32, N>>,
        partials: &mut Partials<N>,
        #[comptime] update: RowUpdate,
        #[comptime] read: RowRead,
    ) {
        let width = N::value();
        let (rows, columns) = comptime!((self.shape.rows(), self.shape.columns()));
        let parts = comptime!(columns / width);

        let key_lines = keys.flat::<f32, N>();
        let query_lines = queries.flat::<f32, N>();
        let factor_all = Vector::<f32, N>::cast_from(factor);

        partials.clear();
        #[unroll]
        for chunk in 0..comptime!(rows / width) {
            let key = key_lines.read(chunk);
            let read_line = match read {
                RowRead::Keys => key,
                RowRead::Queries => query_lines.read(chunk),
            };
            #[unroll]
            for lane in 0..width {
                let row = chunk * width + lane;
                let k_all = Vector::<f32, N>::cast_from(key.extract(lane));
                let read_all = match read {
                    RowRead::Keys => k_all,
                    RowRead::Queries => Vector::<f32, N>::cast_from(read_line.extract(lane)),
                };
                #[unroll]
                for part in 0..parts {
                    let line = self.lines[row * parts + part];
                    let cell = match update {
                        RowUpdate::Decay => line * factor_all,
                        RowUpdate::Write => line + k_all * write[part],
                        RowUpdate::DecayWrite => line * factor_all + k_all * write[part],
                    };
                    self.lines[row * parts + part] = cell;
                    partials.add(part, cell * read_all);
                }
            }
        }
    }

    /// The rows written as kept state `index` of `kept`, windowed to the unit: its states one
    /// after another, all the unit's rows of each.
    pub fn store(&self, kept: &mut Tile, index: usize) {
        let width = N::value();
        let (rows, columns) = comptime!((self.shape.rows(), self.shape.columns()));
        let parts = comptime!(columns / width);

        let mut writes = kept.flat_mut::<f32, N>();
        #[unroll]
        for line in 0..(rows * parts) {
            writes.write(index * (rows * parts) + line, self.lines[line]);
        }
    }
}
