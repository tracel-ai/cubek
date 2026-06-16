pub mod schedule;

use cubecl::prelude::*;

use crate::{
    Linear, Partitioner, Region, Schedule, Tile, TileExpand,
    quantization::dequantize::schedule::dequantize_direct,
};

#[cube]
impl<O: CubePrimitive> Tile<O> {
    /// naive implementation only for per tensor native
    pub fn dequantize<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        input: &Tile<I>,
        scales: &Tile<S>,
    ) where
        O: Dequantize<I, S>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => Dequantize::dequantize(input, scales, self),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => dequantize_direct(input, scales, self),
                _ => {
                    unimplemented!(
                        "currently unsupported schedule: {:?}. only {:?} is supported",
                        level.schedule(),
                        Schedule::Direct
                    );
                }
            },
        }
    }

    pub fn dequantize_at<I: CubePrimitive, S: CubePrimitive>(
        &mut self,
        input: &Tile<I>,
        scales: &Tile<S>,
        region: &Region,
    ) where
        O: Dequantize<I, S>,
    {
        self.at(region).dequantize(&input.at(region), scales);
    }
}

#[cube]
pub trait Dequantize<I: CubePrimitive, S: CubePrimitive>: CubePrimitive {
    fn dequantize(input: &Tile<I>, scales: &Tile<S>, output: &mut Tile<Self>);
}

#[cube]
impl<I: Numeric, S: Numeric, O: Numeric, IN: Size, SN: Size, ON: Size>
    Dequantize<Vector<I, IN>, Vector<S, SN>> for Vector<O, ON>
{
    fn dequantize(
        input: &Tile<Vector<I, IN>>,
        scales: &Tile<Vector<S, SN>>,
        output: &mut Tile<Vector<O, ON>>,
    ) {
        // per-tensor: one scale at flat position 0
        let scale = Vector::cast_from(scales.view().read(seq![0]));

        // Re-view both operands as flat 1-D: `Linear` turns the index into the N-D position,
        // so the leaf scans linearly without re-deriving strides or the element count.
        let input_view = input.view();
        let input_shape = input_view.shape();
        let input_view = input_view.view(Linear::new(input_shape.clone()));
        let mut out = output.view_mut().view_mut(Linear::new(input_shape.clone()));

        for i in 0..out.shape() {
            out.write(i, Vector::cast_from(input_view.read(i)) * scale);
        }
    }
}
