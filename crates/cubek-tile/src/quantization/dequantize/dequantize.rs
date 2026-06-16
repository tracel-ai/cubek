use cubecl::prelude::*;

use crate::{
    MemData, Partitioner, Payload, PayloadExpand, Region, Schedule, Space, Tile, TileExpand, Walk,
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
pub(crate) fn dequantize_direct<
    I: CubePrimitive,
    S: CubePrimitive,
    O: CubePrimitive + Dequantize<I, S>,
>(
    input: &Tile<I>,
    scales: &Tile<S>,
    output: &mut Tile<O>,
) {
    let space = comptime![Space::merge(&[&input.space, &output.space])];
    for region in Walk::over(space) {
        output.dequantize_at(input, scales, &region);
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
        let space = comptime![input.space.clone()];
        let payload = &mut output.payload;
        match payload {
            Payload::Cmma(_) => {
                panic!("Dequantize: cmma fragment has no memory view")
            }
            Payload::Gmem(g) | Payload::Smem(g) => {
                dequantize_register_memory::<I, S, O, IN, SN, ON>(g, input, scales, space)
            }
        };
    }
}

#[cube]
fn dequantize_register_memory<I: Numeric, S: Numeric, O: Numeric, IN: Size, SN: Size, ON: Size>(
    output: &mut MemData<Vector<O, ON>>,
    input: &Tile<Vector<I, IN>>,
    scales: &Tile<Vector<S, SN>>,
    #[comptime] space: Space,
) {
    let scale = scales.view().read(seq![0]);
    let scale = Vector::<O, ON>::cast_from(scale);

    // let matrices = output.matrix_count();
    // for m in 0..matrices {
    //     let v = input.matrix(m);
    //     let mut o = output.matrix_mut(m);

    //     let (h, w) = o.shape();
    //     for r in 0..h {
    //         for c in 0..w {
    //             let q = v.read((r, c));
    //             o.write((r, c), Vector::cast_from(q) * scale);
    //         }
    //     }
    // }
}
