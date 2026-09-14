//! A host model of a barrier slot's protocol, run over every legal interleaving of the two
//! sides. No device: the point is to answer, before a kernel is ever launched, the question a
//! device answers by hanging.
//!
//! The model is the acquire/release pairs of [`Staging`](crate::Staging) read as a state
//! machine. A producer unit waits `empty` on the parity its `writes` was not born at, fills if
//! it is the elected one, arrives `full`, and flips. A consumer unit waits `full` on its
//! `reads`, reads, arrives `empty`, and flips. Both walk the same regions, slot
//! `region % depth`, and neither knows where the other is.
//!
//! What it checks is what a wrong arrival count does: nobody reads a slot before it holds its
//! region, nobody refills one while a reader is still in it, both sides take exactly as many
//! steps as there are regions, each slot's two parities end where they started, and — the one
//! that matters — no reachable state leaves every unfinished unit waiting.

use std::collections::HashSet;

/// One mbarrier: the arrivals that complete a phase, how many have come, and the parity of the
/// phase in progress. A wait on the parity in progress blocks; any other parity passes. That is
/// why a producer's first wait, on the parity its counter was *not* born at, goes straight
/// through a barrier nobody has arrived on.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Mbarrier {
    arrivals: u32,
    arrived: u32,
    parity: u32,
}

impl Mbarrier {
    fn new(arrivals: u32) -> Mbarrier {
        Mbarrier {
            arrivals,
            arrived: 0,
            parity: 0,
        }
    }

    fn arrive(&mut self) {
        self.arrived += 1;
        if self.arrived == self.arrivals {
            self.arrived = 0;
            self.parity ^= 1;
        }
    }

    fn passes(&self, waited: u32) -> bool {
        self.parity != waited
    }
}

/// One slot: its two barriers, the region its buffer holds, and how many consumers are in it.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Slot {
    full: Mbarrier,
    empty: Mbarrier,
    /// The region the buffer was filled with, `None` before any fill.
    holds: Option<usize>,
    readers: u32,
}

/// One unit's place in its side's walk: the region it is on, and whether it is still at the wait.
///
/// The parity it waits with is not here because it is not the unit's: each slot owns a parity per
/// side, flipped by that side's release, so a unit waiting at slot `s` waits with the number of
/// laps it has already made around the ring.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Unit {
    step: usize,
    waiting: bool,
}

impl Unit {
    fn new() -> Unit {
        Unit {
            step: 0,
            waiting: true,
        }
    }

    /// The parity this unit's next wait carries: one flip per lap of the ring.
    fn parity(&self, depth: usize) -> u32 {
        ((self.step / depth) % 2) as u32
    }
}

/// The protocol's whole state: the ring's slots, the units of each side, and the walk's length.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct State {
    slots: Vec<Slot>,
    producers: Vec<Unit>,
    consumers: Vec<Unit>,
}

/// The shape of one run: how deep the ring is, how many regions the walk has, and how many units
/// stand on each side. The first producer is the elected one, the only unit that writes; the
/// others fill nothing and arrive only to keep step.
#[derive(Clone, Copy, Debug)]
struct Shape {
    depth: usize,
    regions: usize,
    producers: u32,
    consumers: u32,
    /// Whether `full` counts every producer's arrival or the elected unit's alone, which is the
    /// choice [`Pipeline::producers`](crate::Pipeline::producers) makes.
    publishes_all: bool,
}

impl Shape {
    /// The arrivals `full` is armed with.
    fn publishers(&self) -> u32 {
        if self.publishes_all {
            self.producers
        } else {
            1
        }
    }

    fn start(&self) -> State {
        State {
            slots: (0..self.depth)
                .map(|_| Slot {
                    full: Mbarrier::new(self.publishers()),
                    empty: Mbarrier::new(self.consumers),
                    holds: None,
                    readers: 0,
                })
                .collect(),
            producers: (0..self.producers).map(|_| Unit::new()).collect(),
            consumers: (0..self.consumers).map(|_| Unit::new()).collect(),
        }
    }

    fn done(&self, state: &State) -> bool {
        let walked = |units: &Vec<Unit>| units.iter().all(|u| u.step == self.regions && u.waiting);
        walked(&state.producers) && walked(&state.consumers)
    }

    /// Walk every reachable state of this shape, checking the protocol at each. Returns how many
    /// distinct states the two sides can be in, which is the interleaving count the run covered.
    fn explore(&self) -> usize {
        self.explore_from(self.start())
    }

    /// [`explore`](Shape::explore) from a state of the caller's own, so a test can arm a barrier
    /// wrongly and watch what it does.
    fn explore_from(&self, start: State) -> usize {
        let mut seen = HashSet::new();
        let mut stack = vec![start.clone()];
        seen.insert(start);

        while let Some(state) = stack.pop() {
            let next = state.moves(*self);
            if next.is_empty() {
                assert!(
                    self.done(&state),
                    "every unfinished unit is waiting and none can pass: {self:?} deadlocks at \
                     {state:?}"
                );
                for (s, slot) in state.slots.iter().enumerate() {
                    assert_eq!(
                        slot.full.parity, slot.empty.parity,
                        "slot {s} was filled and read a different number of times"
                    );
                    assert_eq!(slot.readers, 0, "slot {s} ends with a reader in it");
                }
                continue;
            }
            for state in next {
                if seen.insert(state.clone()) {
                    stack.push(state);
                }
            }
        }

        seen.len()
    }
}

impl State {
    /// Every move one unit could make from here, as the state it would leave behind. A unit that
    /// has walked every region, or whose wait does not pass, makes none.
    fn moves(&self, shape: Shape) -> Vec<State> {
        let mut next = Vec::new();

        for u in 0..self.producers.len() {
            let unit = self.producers[u].clone();
            if unit.step == shape.regions {
                continue;
            }
            let slot = unit.step % shape.depth;
            let mut after = self.clone();
            if unit.waiting {
                if !after.slots[slot].empty.passes(unit.parity(shape.depth) ^ 1) {
                    continue;
                }
                // The wait passed. The elected unit is the one that writes, so it is the one whose
                // pass has to mean the slot is free; the others fill nothing and only keep step.
                if u == 0 {
                    assert!(
                        after.slots[slot].holds.is_none() || after.slots[slot].readers == 0,
                        "a slot was refilled while a consumer was still reading it"
                    );
                    after.slots[slot].holds = Some(unit.step);
                }
                after.producers[u].waiting = false;
            } else {
                if shape.publishes_all || u == 0 {
                    after.slots[slot].full.arrive();
                }
                after.producers[u].step += 1;
                after.producers[u].waiting = true;
            }
            next.push(after);
        }

        for u in 0..self.consumers.len() {
            let unit = self.consumers[u].clone();
            if unit.step == shape.regions {
                continue;
            }
            let slot = unit.step % shape.depth;
            let mut after = self.clone();
            if unit.waiting {
                if !after.slots[slot].full.passes(unit.parity(shape.depth)) {
                    continue;
                }
                assert_eq!(
                    after.slots[slot].holds,
                    Some(unit.step),
                    "a slot was read holding a region other than the one being walked"
                );
                after.slots[slot].readers += 1;
                after.consumers[u].waiting = false;
            } else {
                after.slots[slot].readers -= 1;
                after.slots[slot].empty.arrive();
                after.consumers[u].step += 1;
                after.consumers[u].waiting = true;
            }
            next.push(after);
        }

        next
    }
}

/// Every ring the two sides can be run over, from a single slot to one deeper than the walk.
#[test]
fn the_two_sides_agree_however_they_interleave() {
    for depth in 1..=3 {
        for regions in 1..=4 {
            for consumers in 1..=3 {
                let shape = Shape {
                    depth,
                    regions,
                    producers: 2,
                    consumers,
                    publishes_all: true,
                };
                assert!(shape.explore() > 0);
            }
        }
    }
}

/// A ring of one slot admits no overlap at all: the producer and the consumer alternate, and
/// each region is filled, read, and freed before the next is touched.
#[test]
fn a_single_slot_never_runs_ahead() {
    let shape = Shape {
        depth: 1,
        regions: 3,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    shape.explore();
}

/// The producer may run `depth - 1` regions ahead and no further, which is the whole point of
/// the ring: the wait on `empty` is what stops it.
#[test]
fn the_producer_runs_at_most_a_ring_ahead() {
    let shape = Shape {
        depth: 2,
        regions: 4,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    let mut seen = HashSet::new();
    let mut stack = vec![shape.start()];
    seen.insert(shape.start());
    let mut lead = 0;
    while let Some(state) = stack.pop() {
        lead = lead.max(state.producers[0].step - state.consumers[0].step);
        for state in state.moves(shape) {
            if seen.insert(state.clone()) {
                stack.push(state);
            }
        }
    }
    assert_eq!(lead, shape.depth);
}

/// The count `empty` is armed with is the one thing the two sides cannot disagree about. Armed
/// for the whole cube rather than the units that read, it never completes and the producer waits
/// forever on a slot every consumer has already freed.
#[test]
#[should_panic(expected = "deadlocks")]
fn a_slot_freed_by_fewer_units_than_it_waits_for_hangs() {
    let shape = Shape {
        depth: 1,
        regions: 2,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    let mut start = shape.start();
    // As if `empty` had been armed with `CUBE_DIM` while only the planes that compute arrive.
    start.slots[0].empty.arrivals = 2;
    shape.explore_from(start);
}

/// And why `full` counts every producer. Published by the elected unit alone, a filling plane
/// that has not yet reached its first wait finds `empty` already flipped by the consumers, waits
/// for a parity that has gone by, and never moves again. Counting its arrival is what holds the
/// window open until it gets there.
#[test]
#[should_panic(expected = "deadlocks")]
fn a_slot_published_by_the_elected_unit_alone_lets_a_second_filling_plane_drift() {
    let shape = Shape {
        depth: 1,
        regions: 1,
        producers: 2,
        consumers: 1,
        publishes_all: false,
    };
    shape.explore();
}
