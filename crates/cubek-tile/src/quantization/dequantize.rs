use cubecl::prelude::*;

use crate::{Tile, TileExpand, Walk};

#[cube]
impl<E: Numeric, N: Size> Tile<Vector<E, N>> {
    /// naive implementation only for per tensor native
    pub fn dequantize<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        values: &Tile<I>,
        scales: &Tile<S>,
    ) {
        // per-tensor: one scale at flat position 0
        let scale = scales.view().read(seq![0u32]);
        let scale = Vector::cast_from(scale);

        for region in Walk::over(values.runtime_space()) {
            let lhs = values.at(&region);
            let mut out = self.at(&region);

            let matrices = out.matrix_count();
            for m in 0..matrices {
                let v = lhs.matrix(m);
                let mut o = out.matrix_mut(m);

                let (h, w) = o.shape();
                for r in 0..h {
                    for c in 0..w {
                        let q = v.read((r, c));
                        o.write((r, c), Vector::cast_from(q) * scale);
                    }
                }
            }
        }
    }
}
