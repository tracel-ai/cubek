use cubecl::prelude::*;
use cubecl::std::tensor::layout::CoordsDyn;

use crate::{Space, Tile, TileExpand, Walk};

#[cube]
impl<O: CubeMul + Cast> Tile<O> {
    /// naive implementation only for per tensor native
    pub fn dequantize<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        values: &Tile<I>,
        scales: &Tile<S>,
    ) {
        let space = comptime!(Space::merge(&[&values.space, &scales.space, &self.space]));
        let walk = Walk::over(space);
        for i in 0..walk.total() {
            let region = walk.region(i);

            let lhs = values.at(&region);
            let rhs = scales.at(&region);
            let mut out = self.at(&region);

            let matrices = out.matrix_count();
            for m in 0..matrices {
                let v = lhs.matrix(m);
                let s = rhs.matrix(m);
                let scale = s.read((0u32, 0u32).runtime()); // per tensor works like this since only one value is needed
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

#[cube]
impl<O: CubeMul + Cast> Tile<O> {
    /// native per-tensor and per-block dequantization (1-D or 2-D blocks).
    ///
    /// Scales are a 2-D block *grid*: one scale per block, laid out
    /// `(m / block_m) × (n / block_n)`. The per-axis block sizes are derived from how
    /// much coarser the scale grid is than the values, so per-tensor (`{1,1}` grid),
    /// 1-D blocks (`{m, n/block_n}` grid) and 2-D blocks all use the same code.
    pub fn dequantize2<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        values: &Tile<I>,
        scales: &Tile<S>,
    ) {
        let space = comptime!(values.space.clone());

        let axis_m = comptime!(space.axis_at(0));
        let axis_n = comptime!(space.axis_at(1));
        let edge_m = comptime!(space.partitioner().edge(axis_m)); // tile edge
        let edge_n = comptime!(space.partitioner().edge(axis_n));

        // Per-axis block size: values extent / scale-grid extent on each axis.
        let block_m = comptime!(space.extent_at(0) / scales.space.extent_at(0));
        let block_n = comptime!(space.extent_at(1) / scales.space.extent_at(1));

        let s_all = scales.matrix(0); // the block grid: (m / block_m) × (n / block_n)

        let walk = Walk::over(comptime!(space.clone()));
        for i in 0..walk.total() {
            let region = walk.region(i);
            let tr = region.coord(axis_m); // tile index (runtime)
            let tc = region.coord(axis_n);
            let lhs = values.at(&region);
            let mut out = self.at(&region);
            let matrices = out.matrix_count();
            for mm in 0..matrices {
                let v = lhs.matrix(mm);
                let mut o = out.matrix_mut(mm);
                let (h, w) = o.shape();
                for r in 0..h {
                    for c in 0..w {
                        let gr = tr * edge_m + r as usize; // global row
                        let gc = tc * edge_n + c as usize; // global col
                        let br = gr / block_m; // block row
                        let bc = gc / block_n; // block col
                        let scale = s_all.read((br as u32, bc as u32));
                        let q = v.read((r, c));
                        o.write((r, c), O::cast_from(q) * O::cast_from(scale));
                    }
                }
            }
        }
    }
}

#[cube]
impl<O: CubeMul + Cast> Tile<O> {
    /// Native per-tensor and per-block dequantization for blocks of any rank.
    ///
    /// Scales are an N-D block *grid*: one scale per block, with the grid `extent[a] /
    /// block_size[a]` on each axis. The per-axis block size is derived from how much
    /// coarser the scale grid is than the values, so per-tensor, 1-D, 2-D and N-D blocks
    /// all share this code. Each value element maps to a block-grid cell by
    /// `cell[a] = global[a] / block_size[a]`.
    pub fn dequantize3<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        values: &Tile<I>,
        scales: &Tile<S>,
    ) {
        let space = comptime!(values.space.clone());
        let rank = comptime!(space.rank());

        let walk = Walk::over(comptime!(space.clone()));
        for i in 0..walk.total() {
            let region = walk.region(i);

            let lhs = values.at(&region);
            let mut out = self.at(&region);
            // The whole block grid, read at absolute cell coords (scales aren't tiled).
            let s_all = scales.view();

            let v = lhs.view();
            let mut o = out.view_mut();
            let shape = v.shape(); // this tile's extent, per axis

            let mut total = 1u32;
            #[unroll]
            for a in 0..rank {
                total *= shape[a];
            }

            for lin in 0..total {
                // Unravel the linear index into local (in-tile) coords, row-major.
                let mut local = CoordsDyn::new();
                #[unroll]
                for a in 0..rank {
                    let mut weight = 1u32;
                    #[unroll]
                    for b in a + 1..rank {
                        weight *= shape[b];
                    }
                    local.push((lin / weight) % shape[a]);
                }

                // Map each axis's global position to its block-grid cell.
                let mut cell = CoordsDyn::new();
                #[unroll]
                for a in 0..rank {
                    let axis = comptime!(space.axis_at(a));
                    let edge = comptime!(space.partitioner().edge(axis));
                    let block = comptime!(space.extent_at(a) / scales.space.extent_at(a));
                    let global = region.coord(axis) * edge + local[a] as usize;
                    cell.push((global / block) as u32);
                }

                let q = v.read(local.clone());
                let scale = s_all.read(cell);
                o.write(local, O::cast_from(q) * O::cast_from(scale));
            }
        }
    }
}
