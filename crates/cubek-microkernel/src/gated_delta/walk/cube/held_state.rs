use super::base::CubeGatedDeltaWalkMicrokernel;
use super::unit_partials::UnitPartials;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// One unit's key rows of its run's state, a line of the run's columns a part of a row, in
/// registers for the whole walk. Every tile it reads or writes is windowed to the unit, so rows
/// the head leaves past its last read as zeros and are never written.
#[derive(CubeType)]
pub struct HeldState<N: Size> {
    lines: Array<Vector<f32, N>>,
    #[cube(comptime)]
    microkernel: CubeGatedDeltaWalkMicrokernel,
}

#[cube]
impl<N: Size> HeldState<N> {
    /// The unit's rows of `state`, windowed to the unit: the state read once.
    pub fn load(state: &Tile, #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel) -> Self {
        let width = N::value();
        let (rows, columns) = comptime!((microkernel.rows, microkernel.columns));
        let parts = comptime!(columns / width);
        let reads = state.flat::<f32, N>();
        let mut lines = Array::<Vector<f32, N>>::new(comptime!(rows * parts));
        #[unroll]
        for line in 0..comptime!(rows * parts) {
            lines[line] = reads.read(line);
        }
        HeldState::<N> { lines, microkernel }
    }

    /// The rows scaled by `factor`, and this unit's part of what they recall of the position's key
    /// (`Σ S[r] · k[r]` over its rows), left in `partials`. `keys` is windowed to the unit and the
    /// position.
    pub fn decay_and_recall(&mut self, keys: &Tile, factor: f32, partials: &mut UnitPartials<N>) {
        let width = N::value();
        let (rows, columns) = comptime!((self.microkernel.rows, self.microkernel.columns));
        let parts = comptime!(columns / width);
        let key_lines = keys.flat::<f32, N>();
        let factor_all = Vector::<f32, N>::cast_from(factor);
        partials.clear();
        #[unroll]
        for chunk in 0..comptime!(rows / width) {
            let key = key_lines.read(chunk);
            #[unroll]
            for lane in 0..width {
                let row = chunk * width + lane;
                let k_all = Vector::<f32, N>::cast_from(key.extract(lane));
                #[unroll]
                for part in 0..parts {
                    let scaled = self.lines[row * parts + part] * factor_all;
                    self.lines[row * parts + part] = scaled;
                    partials.add(part, scaled * k_all);
                }
            }
        }
    }

    /// `write` added along each row's key, and this unit's part of what the written rows answer to
    /// the position's query (`Σ S[r] · q[r]`), left in `partials`. `keys` and `queries` are
    /// windowed to the unit and the position.
    pub fn write_and_answer(
        &mut self,
        keys: &Tile,
        queries: &Tile,
        write: &Array<Vector<f32, N>>,
        partials: &mut UnitPartials<N>,
    ) {
        let width = N::value();
        let (rows, columns) = comptime!((self.microkernel.rows, self.microkernel.columns));
        let parts = comptime!(columns / width);
        let key_lines = keys.flat::<f32, N>();
        let query_lines = queries.flat::<f32, N>();
        partials.clear();
        #[unroll]
        for chunk in 0..comptime!(rows / width) {
            let key = key_lines.read(chunk);
            let query = query_lines.read(chunk);
            #[unroll]
            for lane in 0..width {
                let row = chunk * width + lane;
                let k_all = Vector::<f32, N>::cast_from(key.extract(lane));
                let q_all = Vector::<f32, N>::cast_from(query.extract(lane));
                #[unroll]
                for part in 0..parts {
                    let written = self.lines[row * parts + part] + k_all * write[part];
                    self.lines[row * parts + part] = written;
                    partials.add(part, written * q_all);
                }
            }
        }
    }

    /// The rows written as kept state `index` of `kept`, windowed to the unit: its states one
    /// after another, all the unit's rows of each.
    pub fn store(&self, kept: &mut Tile, index: usize) {
        let width = N::value();
        let (rows, columns) = comptime!((self.microkernel.rows, self.microkernel.columns));
        let parts = comptime!(columns / width);
        let mut writes = kept.flat_mut::<f32, N>();
        #[unroll]
        for line in 0..comptime!(rows * parts) {
            writes.write(index * comptime!(rows * parts) + line, self.lines[line]);
        }
    }
}
