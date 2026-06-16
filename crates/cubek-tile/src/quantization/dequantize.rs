use cubecl::prelude::*;

use crate::{Tile, TileExpand, Walk};

#[cube]
impl<O: CubeMul + Cast> Tile<O> {
    /// naive implementation only for per tensor native
    pub fn dequantize<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        values: &Tile<I>,
        scales: &Tile<S>,
    ) {
        let walk = Walk::over(values.space.clone());
        for i in 0..walk.total() {
            let region = walk.region(i);

            let lhs = values.at(&region);
            let rhs = scales.at(&region);
            let mut out = self.at(&region);

            let matrices = out.matrix_count();
            for m in 0..matrices {
                let v = lhs.matrix(m);
                let s = rhs.matrix(m);
                let scale = s.read((0u32, 0u32).runtime()); // per tensor works since only one scale is needed
                let mut o = out.matrix_mut(m);

                let (h, w) = o.shape();
                for r in 0..h {
                    for c in 0..w {
                        let q = v.read((r, c));
                        o.write((r, c), O::cast_from(q) * O::cast_from(scale));
                    }
                }
            }
        }
    }
}
