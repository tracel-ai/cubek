use super::shape::HeldStateShape;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// What a unit's rows contribute to a sum over its run's units, as lines of the run's columns.
#[derive(CubeType)]
pub struct Partials<N: Size> {
    own: Array<Vector<f32, N>>,
    #[cube(comptime)]
    shape: HeldStateShape,
}

#[cube]
impl<N: Size> Partials<N> {
    /// Partials of `shape`'s columns, holding nothing yet.
    pub fn new(#[comptime] shape: HeldStateShape) -> Self {
        let width = N::value();
        let parts = comptime!(shape.columns() / width);
        Partials::<N> {
            own: Array::<Vector<f32, N>>::new(parts),
            shape,
        }
    }

    /// The partials, zeroed.
    pub fn clear(&mut self) {
        let width = N::value();
        let parts = comptime!(self.shape.columns() / width);
        let zero = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for part in 0..parts {
            self.own[part] = zero;
        }
    }

    /// `value` added to the partial for the columns of line `part`.
    pub fn add(&mut self, part: usize, value: Vector<f32, N>) {
        self.own[part] += value;
    }

    /// The partial for the columns of line `part`: the sum over the run's units once
    /// [`UnitSum::sum`](super::UnitSum::sum) has met.
    pub fn get(&self, part: usize) -> Vector<f32, N> {
        self.own[part]
    }

    /// The partial for the columns of line `part` replaced by `value`.
    pub fn set(&mut self, part: usize, value: Vector<f32, N>) {
        self.own[part] = value;
    }
}
