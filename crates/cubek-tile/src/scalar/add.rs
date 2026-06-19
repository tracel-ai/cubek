use cubecl::prelude::*;

use crate::*;

#[cube]
impl<O: CubePrimitive> Tile<O> {
    pub fn add_scalar<I: CubePrimitive, S: Scalar>(&mut self, input: &Tile<I>, scalar: S)
    where
        O: AddScalar<I, S>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => AddScalar::add(input, scalar, self),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => add_direct(input, scalar, self),
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

    pub fn add_scalar_at<I: CubePrimitive, S: Scalar>(
        &mut self,
        input: &Tile<I>,
        scalar: S,
        region: &Region,
    ) where
        O: AddScalar<I, S>,
    {
        self.at(region).add_scalar(&input.at(region), scalar);
    }
}

#[cube]
pub(crate) fn add_direct<I: CubePrimitive, S: Scalar, O: CubePrimitive + AddScalar<I, S>>(
    input: &Tile<I>,
    scalar: S,
    output: &mut Tile<O>,
) {
    for region in Walk::over(output.runtime_space()) {
        output.add_scalar_at(input, scalar, &region);
    }
}

#[cube]
pub trait AddScalar<I: CubePrimitive, S: Scalar>: CubePrimitive {
    fn add(input: &Tile<I>, scalar: S, output: &mut Tile<Self>);
}

#[cube]
impl<I: Numeric, S: Scalar, O: Numeric, N: Size> AddScalar<Vector<I, N>, S> for Vector<O, N> {
    fn add(input: &Tile<Vector<I, N>>, scalar: S, output: &mut Tile<Vector<O, N>>) {
        let scalar = Vector::cast_from(scalar);

        // Dequant-on-read: a plain tile reads through unchanged, a quantized one folds in its scale.
        let in_flat = input.flat();
        let mut out = output.flat_mut();

        for i in 0..out.shape() {
            out.write(i, Vector::cast_from(in_flat.read(i)) + scalar);
        }
    }
}
