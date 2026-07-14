use cubecl::{
    prelude::*,
    quant::scheme::{QuantLevel, QuantScheme, QuantStore},
    std::tensor::layout::Coords1d,
};

use crate::*;

/// The `Quantized` arm of a [`TileView`]: a flat masked view over the storage element `I` the
/// buffer truly holds, plus the scale geometry to fold each read back into the served `O`. Reads
/// are 1-D ([`Coords1d`], the only shape [`flat`](crate::Tile::flat) yields), so the scale for a
/// line is looked up from its flat position: per-tensor broadcasts the single scale, a block
/// scheme decomposes the position over the window [`extent`](Self::extent) and picks one scale per
/// block. Windows are assumed block-aligned (the tiling invariant), so the block index within the
/// window adds onto the window's own [`window_start`](Self::window_start).
#[derive(CubeType)]
pub struct QuantizedView<'a, O: Numeric, I: Numeric, W: Size> {
    values: FlatView<'a, Vector<I, W>>,
    /// The window-origin scale, resolved: the whole answer for per-tensor (broadcast), and the
    /// block schemes' base at [`window_start`](Self::window_start).
    scale: O,
    /// The whole scales buffer; one scale per block, indexed per line by [`scale_at`](Self::scale_at).
    scales: Box<[f32]>,
    /// Per-axis scale stride (one step per block); `0` on every axis for per-tensor.
    strides: Coords<u32>,
    /// Flat scale index of the window's origin (block-aligned); the per-block advances add onto it.
    window_start: u32,
    /// The window's per-axis extent (inner axis in lines), the radix that decodes a flat position.
    extent: Coords<u32>,
    /// Per-axis block edges (elements per block).
    #[cube(comptime)]
    block: Vec<usize>,
    /// Physical line width, so the inner axis's line position scales back to elements.
    #[cube(comptime)]
    vector_size: usize,
    #[cube(comptime)]
    scheme: QuantScheme,
}

#[cube]
impl<'a, O: Numeric, I: Numeric, W: Size> QuantizedView<'a, O, I, W> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        values: FlatView<'a, Vector<I, W>>,
        scale: O,
        scales: Box<[f32]>,
        strides: Coords<u32>,
        window_start: u32,
        extent: Coords<u32>,
        #[comptime] block: Vec<usize>,
        #[comptime] vector_size: usize,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, O, I, W> {
            values,
            scale,
            scales,
            strides,
            window_start,
            extent,
            block,
            vector_size,
            scheme,
        }
    }

    pub fn read(&self, pos: Coords1d) -> Vector<O, W> {
        let raw = Vector::<O, W>::cast_from(self.values.read(pos));
        match comptime!(self.scheme.store) {
            QuantStore::Native => raw * Vector::cast_from(self.scale_at(pos)),
            _ => panic!("only native quantization storage is supported for now"),
        }
    }

    /// The scale for the line at flat position `pos`. Per-tensor is the single window scale;
    /// a block scheme decodes `pos` over the [`extent`](Self::extent) (row-major, the same radix
    /// [`FlatLayout`](crate::FlatLayout) uses) and dots each axis's block index with its scale
    /// stride, added onto [`window_start`](Self::window_start) (the window is block-aligned, so the
    /// origin's block index folds out into `window_start`).
    fn scale_at(&self, pos: Coords1d) -> O {
        if comptime!(self.scheme.level == QuantLevel::Tensor) {
            self.scale
        } else {
            let rank = self.extent.len().comptime();
            let last = comptime!(rank - 1);
            let mut offs = pos as u32;
            let mut acc = self.window_start;
            // Peel the least-significant axis each step (row-major), carrying the quotient up.
            #[unroll]
            for i in 0..rank {
                let dim = comptime!(rank - 1 - i);
                let extent = self.extent.at(dim);
                let coord = offs % extent;
                offs /= extent;
                // The inner axis's coord is a line index; scale it back to elements.
                let w = comptime!((if dim == last { self.vector_size } else { 1 }) as u32);
                let block = comptime!(self.block[dim] as u32).runtime();
                acc = acc.fadd(
                    coord
                        .fmul(w.runtime())
                        .fdiv(block)
                        .fmul(self.strides.at(dim)),
                );
            }
            O::cast_from(self.scales[acc.fcast::<usize>()])
        }
    }

    pub fn shape(&self) -> Coords1d {
        self.values.shape()
    }
}
