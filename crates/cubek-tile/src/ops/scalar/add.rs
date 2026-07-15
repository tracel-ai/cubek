use cubecl::prelude::*;

use crate::*;

#[cube]
impl<O: Numeric> Tile<O> {
    /// Elementwise `input + scalar` into `self`. Both tiles serve `O`; a quantized input
    /// dequantizes on read ([`Tile::flat`]). `I` is the input's storage element, threaded from
    /// the kernel (unused on a plain input).
    pub fn add_scalar<I: Numeric, S: Scalar>(&mut self, input: &Tile<O>, scalar: S) {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final(_) => add_leaf::<I, S, O>(input, scalar, self),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => add_direct::<I, S, O>(input, scalar, self),
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

    pub fn add_scalar_at<I: Numeric, S: Scalar>(
        &mut self,
        input: &Tile<O>,
        scalar: S,
        region: &Region,
    ) {
        self.at(region)
            .add_scalar::<I, S>(&input.at(region), scalar);
    }
}

#[cube]
pub(crate) fn add_direct<I: Numeric, S: Scalar, O: Numeric>(
    input: &Tile<O>,
    scalar: S,
    output: &mut Tile<O>,
) {
    for region in Walk::over(output.runtime_space()) {
        output.add_scalar_at::<I, S>(input, scalar, &region);
    }
}

#[cube]
fn add_leaf<I: Numeric, S: Scalar, O: Numeric>(input: &Tile<O>, scalar: S, output: &mut Tile<O>) {
    let size!(W) = output.vector_size();
    let scalar = Vector::<O, W>::cast_from(scalar);

    let values = input.flat::<I, W>();
    let mut out = output.flat_mut::<W>();

    for i in 0..out.shape() {
        out.write(i, values.read(i) + scalar);
    }
}
