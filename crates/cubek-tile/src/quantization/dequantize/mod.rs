pub mod schedule;

use cubecl::prelude::*;

use crate::{dequantize::schedule::dequantize_direct, *};

#[cube]
impl<O: Numeric> Tile<O> {
    /// naive implementation only for per tensor native
    pub fn dequantize<I: Numeric, S: Numeric>(&mut self, input: &Tile<I>, scales: &Tile<S>)
    where
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

    pub fn dequantize_at<I: Numeric, S: Numeric>(
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
pub trait Dequantize<I: Numeric, S: Numeric>: Numeric {
    fn dequantize(input: &Tile<I>, scales: &Tile<S>, output: &mut Tile<Self>);
}

#[cube]
impl<I: Numeric, S: Numeric, O: Numeric> Dequantize<I, S> for O {
    fn dequantize(input: &Tile<I>, scales: &Tile<S>, output: &mut Tile<O>) {
        // The physical widths are storage detail; reconstruct each operand's lines. Input and
        // output share the logical shape, so they scan at the same width.
        let size!(W) = output.vector_size();
        let size!(SW) = scales.vector_size();

        // per-tensor: one scale at flat position 0, broadcast across the output line.
        let scale = Vector::<O, W>::cast_from(scales.view::<SW>().read(seq![0]));

        let values = input.flat::<W>();
        let mut out = output.flat_mut::<W>();

        for i in 0..out.shape() {
            out.write(i, Vector::<O, W>::cast_from(values.read(i)) * scale);
        }
    }
}
