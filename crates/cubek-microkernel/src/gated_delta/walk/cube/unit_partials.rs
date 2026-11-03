use super::base::CubeGatedDeltaWalkMicrokernel;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// What a unit's rows contribute to a sum over its run's units, as lines of the run's columns, and
/// the cube's shared memory the units meet them through.
///
/// A sum writes every unit's lines to a slice of the shared memory, meets at one barrier, and has
/// each unit add every unit's lines in unit order. The two sums of a position use the two slices
/// in turn, so a unit still reading one sum is never overwritten by the next.
#[derive(CubeType)]
pub struct UnitPartials<N: Size> {
    shared: Shared<[Vector<f32, N>]>,
    own: Array<Vector<f32, N>>,
    run: usize,
    unit: usize,
    #[cube(comptime)]
    microkernel: CubeGatedDeltaWalkMicrokernel,
    #[cube(comptime)]
    runs: usize,
}

#[cube]
impl<N: Size> UnitPartials<N> {
    /// Room for the units of `runs` runs, holding nothing yet; this unit is unit `unit` of run
    /// `run`.
    pub fn new(
        run: usize,
        unit: usize,
        #[comptime] runs: usize,
        #[comptime] microkernel: CubeGatedDeltaWalkMicrokernel,
    ) -> Self {
        let width = N::value();
        let parts = comptime!(microkernel.columns / width);
        UnitPartials::<N> {
            shared: Shared::<[Vector<f32, N>]>::new_slice(comptime!(
                2 * runs * microkernel.units * parts
            )),
            own: Array::<Vector<f32, N>>::new(parts),
            run,
            unit,
            microkernel,
            runs,
        }
    }

    /// This unit's partials, zeroed.
    pub fn clear(&mut self) {
        let width = N::value();
        let parts = comptime!(self.microkernel.columns / width);
        let zero = Vector::<f32, N>::cast_from(0.0f32);
        #[unroll]
        for part in 0..parts {
            self.own[part] = zero;
        }
    }

    /// `value` added to this unit's partial for the columns of line `part`.
    pub fn add(&mut self, part: usize, value: Vector<f32, N>) {
        self.own[part] += value;
    }

    /// The partial for the columns of line `part`: the sum over the run's units once
    /// [`sum`](Self::sum) has met.
    pub fn get(&self, part: usize) -> Vector<f32, N> {
        self.own[part]
    }

    /// Every partial summed over the run's units, in unit order, through `slice` of the shared
    /// memory.
    pub fn sum(&mut self, #[comptime] slice: usize) {
        let units = comptime!(self.microkernel.units);
        if comptime!(units > 1) {
            let width = N::value();
            let parts = comptime!(self.microkernel.columns / width);
            let mine = self.slot(slice, self.unit);
            #[unroll]
            for part in 0..parts {
                self.shared[mine + part] = self.own[part];
            }
            sync_cube();
            let head = self.slot(slice, 0usize);
            #[unroll]
            for part in 0..parts {
                self.own[part] = self.shared[head + part];
            }
            for other in 1..units {
                let at = self.slot(slice, other);
                #[unroll]
                for part in 0..parts {
                    self.own[part] += self.shared[at + part];
                }
            }
        }
    }

    /// Where `unit`'s lines of `slice` start in the shared memory.
    fn slot(&self, #[comptime] slice: usize, unit: usize) -> usize {
        let width = N::value();
        let units = comptime!(self.microkernel.units);
        let parts = comptime!(self.microkernel.columns / width);
        ((slice * self.runs + self.run) * units + unit) * parts
    }
}
