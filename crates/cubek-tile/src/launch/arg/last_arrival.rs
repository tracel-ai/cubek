//! A box the runs of a share hold parts of, merged by the last run to arrive: the [`LastArrival`].

use cubecl::prelude::*;

use crate::launch::arg::relay::folding_tile;
use crate::launch::arg::tensor::TileArgExpand;
use crate::space::Extent;
use crate::*;

/// One output box of a [`Levels::shared_by`] share, held whole by one run or in parts by several
/// ([`Portion::holders`]). A run holding it whole hands its sum straight on to the output. Each run
/// holding a part parks it in a slot of its own and counts itself in at the box; the last to count
/// adds the box's slots in the order of their runs and hands the total on to the output.
///
/// No run waits on another, so the merge does not depend on how the device schedules cubes. The
/// parts are added in the order of their runs, so the total is the same bits whichever run arrives
/// last.
///
/// The slots are one operand of two boxes a run ([`Partitioning::arrival_slots`]): the first holds
/// the part of the box the run starts in, the second the part of the box it ends in. A run's other
/// boxes are its own whole, so no part needs a third. The counters are one `u32` a box
/// ([`Partitioning::arrival_counters`]), holding zero, and left holding zero. A count publishes the
/// run's part at device scope first, so it runs only where the runtime hands one cube's writes to
/// another within a dispatch (`device_memory_scope`).
#[derive(CubeType)]
pub struct LastArrival<'a, E: Numeric> {
    /// The slots, as the boxes of [`Partitioning::arrival_slots`]'s space.
    slots: Tile<E>,
    /// The slots written adding into what they hold, as the last run sums into the first's.
    folding: Tile<E>,
    /// The slot boxes, one flat index: a run's two are `2 · run` and the next.
    boxes: Walk,
    /// This run's slot of this box.
    own: Tile<E>,
    arrivals: &'a [Atomic<u32>],
    /// This run's box of the output.
    region: Region,
    /// This box's counter.
    counter: usize,
    /// The runs holding parts of this box, from the first.
    first: usize,
    holders: usize,
    /// The first run's slot of this box: its second where it started in a box before.
    first_slot: usize,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// This operand as the slots of a [`LastArrival`] at the `i`-th box `portion` touches, under
    /// `partitioning`, whose cube level shares its steps ([`shared_by`](Levels::shared_by)). The
    /// operand is laid out as [`Partitioning::arrival_slots`] says for its axes.
    pub fn last_arrival<'t>(
        &self,
        partitioning: &Partitioning,
        portion: &Portion,
        i: usize,
        arrivals: &'t [Atomic<u32>],
    ) -> LastArrival<'t, E> {
        let boxes_space = comptime!(partitioning.clone().arrival_slots_space(self.spec.axes()));
        let boxes = Partitioning {
            space: Space::with_sizes(
                comptime!(boxes_space.space().clone()),
                partitioning.space.sizes.clone(),
            ),
            levels: comptime!(boxes_space.levels().to_vec()),
        };
        let (first, holders) = portion.holders(i);
        let run = portion.run();
        let first_slot = 2 * first + select(portion.starts_in(i, first), 0usize, 1usize);
        let slots = self.tile(comptime!(boxes_space.clone()));
        let walk = boxes.walk();
        let own_box = walk.region(2 * run + select(portion.starts_in(i, run), 0usize, 1usize));
        let own = slots.at(&own_box);
        LastArrival::<'t, E> {
            slots,
            folding: folding_tile::<E, V>(self, &boxes, false),
            boxes: walk,
            own,
            arrivals,
            region: portion.region(i),
            counter: portion.index(i),
            first,
            holders,
            first_slot,
        }
    }
}

#[cube]
impl<'a, E: Numeric> LastArrival<'a, E> {
    /// This run's slot of its box: what a sum it hands on is opened on.
    pub fn tile(&self) -> Tile<E> {
        self.own.clone()
    }

    /// Hand `sum`, opened on [`tile`](LastArrival::tile), on to `out` with a cast: drained straight
    /// into it where this run holds the box whole, parked and merged by the last run to arrive
    /// otherwise. Every unit of the cube reaches it.
    pub fn hand_on<O: Numeric>(&self, sum: &Tile<E>, out: &mut Tile<O>) {
        if self.holders == 1 {
            sum.drained_into(&out.at(&self.region));
        } else {
            sum.drained_into(&self.tile());
            // The part lands at device scope before the run counts itself in.
            sync_storage();
            let mut arrived = Shared::<[u32]>::new_slice(1usize);
            if UNIT_POS == 0 {
                arrived[0] = self.arrivals[self.counter].fetch_add(1u32);
            }
            sync_cube();
            // Uniform across the cube: only the last run merges.
            if arrived[0] as usize + 1 == self.holders {
                // What the other runs parked is acquired before it is read.
                sync_storage();
                let total = self.boxes.region(self.first_slot);
                let mut sum_into = self.folding.at(&total);
                // Every run after the first starts in this box: its part is in its first slot.
                for run in self.first + 1..self.first + self.holders {
                    sum_into.copy_from(&self.slots.at(&self.boxes.region(2 * run)));
                }
                // Every line of the total is in before any is read back.
                sync_storage();
                let mut out_box = out.at(&self.region);
                out_box.copy_cast_from(&self.slots.at(&total));
                if UNIT_POS == 0 {
                    self.arrivals[self.counter].store(0u32);
                }
            }
        }
    }
}

impl Partitioning {
    /// The arrival counters a [`LastArrival`] under this partitioning counts its runs in through:
    /// one `u32` a box of its shared cube level, holding zero.
    pub fn arrival_counters(&self) -> usize {
        let level = &self.levels()[0];
        self.space()
            .axes()
            .map(|axis| level.tiles(self.space(), axis))
            .product()
    }

    /// The shape of a [`LastArrival`]'s slots for an output over `axes`, in that order: two cube
    /// boxes a run of the shared cube level, stacked along the first of `axes`.
    pub fn arrival_slots(&self, axes: &[Axis]) -> cubecl::zspace::Shape {
        let boxes = self.arrival_slots_space(axes);
        cubecl::zspace::Shape::from(axes.iter().map(|&axis| boxes.space().extent(axis)))
    }

    /// The space the slots of a [`LastArrival`] for an output over `axes` are boxes of, cut by
    /// these levels: this space with every axis of `axes` one cube box long, the first stacking
    /// two boxes a run. The other axes keep their extents, so the cube level cuts them as it cuts
    /// the output's, which spans each in one box.
    fn arrival_slots_space(&self, axes: &[Axis]) -> Partitioning {
        let level = &self.levels()[0];
        let runs = level.shared_by().expect(
            "LastArrival: the cube level distributes no grid as one index; say `shared_by`",
        );
        let cube_box = level.child(self.space());
        let extents: Vec<(Axis, Extent)> = self
            .space()
            .axes()
            .map(|axis| match axes.iter().position(|&a| a == axis) {
                None => (axis, self.space().extent_raw(axis)),
                Some(at) => {
                    let Extent::Static(edge) = cube_box.extent_raw(axis) else {
                        panic!(
                            "LastArrival: the cube box's {axis:?} extent is known only at launch; \
                             a slot is one cube box, so the cube level cuts the output's axes by a \
                             stated tile"
                        )
                    };
                    let stacked = if at == 0 { 2 * runs } else { 1 };
                    (axis, Extent::Static(stacked * edge))
                }
            })
            .collect();
        Partitioning::new(Space::from_extents(&extents), self.levels().to_vec())
    }
}
