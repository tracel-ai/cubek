use super::partials::Partials;
use super::shape::HeldStateShape;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// The sum of each unit's [`Partials`] over a run's units, met through the cube's shared memory.
///
/// A sum writes every unit's lines to one of two slices of the shared memory, meets at one barrier,
/// and has each unit add every unit's lines in unit order. Consecutive sums use the two slices in
/// turn, which `UnitMeet` alone picks, so a unit still reading one sum is never overwritten by the
/// next.
#[derive(CubeType)]
pub struct UnitMeet<N: Size> {
    shared: Shared<[Vector<f32, N>]>,
    run: usize,
    unit: usize,
    next: usize,
    #[cube(comptime)]
    shape: HeldStateShape,
    #[cube(comptime)]
    runs: usize,
}

#[cube]
impl<N: Size> UnitMeet<N> {
    /// Room for the units of `runs` runs; this unit is unit `unit` of run `run`.
    pub fn new(
        run: usize,
        unit: usize,
        #[comptime] runs: usize,
        #[comptime] shape: HeldStateShape,
    ) -> Self {
        let width = N::value();
        let parts = comptime!(shape.columns() / width);
        UnitMeet::<N> {
            shared: Shared::<[Vector<f32, N>]>::new_slice(2 * runs * shape.units() * parts),
            run,
            unit,
            next: 0usize,
            shape,
            runs,
        }
    }

    /// Every partial of `partials` summed over the run's units, in unit order, through the slice of
    /// the shared memory the last sum did not use.
    pub fn sum(&mut self, partials: &mut Partials<N>) {
        let units = comptime!(self.shape.units());
        if units > 1 {
            let slice = self.next;
            self.next = 1usize - slice;
            let width = N::value();
            let parts = comptime!(self.shape.columns() / width);

            let mine = self.slot(slice, self.unit);
            #[unroll]
            for part in 0..parts {
                self.shared[mine + part] = partials.get(part);
            }

            sync_cube();

            let head = self.slot(slice, 0usize);
            #[unroll]
            for part in 0..parts {
                partials.set(part, self.shared[head + part]);
            }

            for other in 1..units {
                let at = self.slot(slice, other);
                #[unroll]
                for part in 0..parts {
                    partials.add(part, self.shared[at + part]);
                }
            }
        }
    }

    /// Where `unit`'s lines of `slice` start in the shared memory.
    fn slot(&self, slice: usize, unit: usize) -> usize {
        let width = N::value();
        let units = comptime!(self.shape.units());
        let parts = comptime!(self.shape.columns() / width);
        ((slice * self.runs + self.run) * units + unit) * parts
    }
}
