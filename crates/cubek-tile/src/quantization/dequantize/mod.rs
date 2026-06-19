pub mod schedule;

use cubecl::prelude::*;

use crate::{dequantize::schedule::dequantize_direct, *};

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

        let values = input.flat();
        let mut out = output.flat_mut();

        for i in 0..out.shape() {
            out.write(i, Vector::cast_from(values.read(i)) * scale);
        }
    }
}
