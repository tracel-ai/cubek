use cubecl::client::Client;
use cubecl::tile::{Axis, Launcher, Levels, Partitioning, Space};
use cubek_std::routine::Grid;

/// A sequence of the walk, or a row of a decode step.
pub const BATCH: Axis = Axis(0);
/// A position of the sequence: the axis the recurrence walks, one step at a time.
pub const POSITION: Axis = Axis(1);
/// A key head, which `heads_per_group` value heads read: a selective state space model's group.
pub const KEY_HEAD: Axis = Axis(2);
/// A value head's place among the `heads_per_group` heads reading one key head: value head
/// `key_head * heads_per_group + place`.
pub const HEAD_IN_GROUP: Axis = Axis(3);
/// A state the walk keeps for the caller to rewind to: the state after each of its last positions,
/// oldest first.
pub const KEPT_STATE: Axis = Axis(4);
/// A state row: a key feature, which the queries and keys run along.
pub const KEY_FEATURE: Axis = Axis(5);
/// A state column: a value feature, which the values and the output run along.
pub const VALUE_FEATURE: Axis = Axis(6);
/// A slot of a store that holds many sequences' states, which a table places each sequence at: no
/// level cuts it.
pub const STORE_SLOT: Axis = Axis(7);
/// A channel the conv mixes.
pub const CHANNEL: Axis = Axis(8);
/// A tap of the conv's filter.
pub const TAP: Axis = Axis(9);
/// A column of a store's ring, the conv's past values of every channel.
pub const RING_COLUMN: Axis = Axis(10);
/// A position of the input that later positions read by tap: no level cuts it.
pub const PROMPT_POSITION: Axis = Axis(11);
/// The two rates of a position and value head, the log of its decay at 0 and its strength at 1: no
/// level cuts it.
pub const RATE: Axis = Axis(12);
/// A run of channel lines one cube works on: no operand binds it, only the cube level cuts it.
pub const CHANNEL_RUN: Axis = Axis(13);
/// A line of channels inside a cube's run, run local: no operand binds it, only the units level
/// cuts it.
pub const LINE_IN_RUN: Axis = Axis(14);
/// One of the last `taps - 1` positions of a prompt, which the write back copies into the ring: no
/// operand binds it, only the cube level cuts it.
pub const RING_POSITION: Axis = Axis(15);

/// The axes the walk's cubes ride: the boxes of value columns on the grid's `x`, the value heads
/// of a key head on its `y`, the key heads on its `z`.
const WALK_CUBES: [Axis; 3] = [VALUE_FEATURE, HEAD_IN_GROUP, KEY_HEAD];

/// The axes the walk leaves to the launch, so a prompt of any length, a decode step of any width
/// and a store of any size run one kernel; the head dims and the kept states are compiled in,
/// since every owner sizes its registers by the first and the cache's ring fixes the second.
const WALK_DYNAMIC: [Axis; 3] = [BATCH, POSITION, STORE_SLOT];

/// The axes a conv leaves to the launch: rows, positions, slots, ring and prompt.
const CONV_DYNAMIC: [Axis; 5] = [BATCH, POSITION, STORE_SLOT, RING_COLUMN, PROMPT_POSITION];

/// How a recurrence kernel places its microkernel's work: the `cubes` axes ride the grid, the
/// first on its `x`, rows come one a cube, and `dynamic` axes are left to the launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrencePartitioning {
    partitioning: Partitioning,
    dynamic: &'static [Axis],
}

impl RecurrencePartitioning {
    /// The walk's placement of `space`, with `inside` a cube's box: a cube for every box of value
    /// columns of one head of one sequence.
    pub fn walk(space: Space, inside: Levels) -> Self {
        Self::new(space, inside, &WALK_CUBES, &WALK_DYNAMIC)
    }

    /// The conv's placement of `space`, with `inside` a cube's run: a cube for every run of
    /// channels of every row, and of every `cubes` axis the routine puts on the grid.
    pub fn conv(space: Space, inside: Levels, cubes: &[Axis]) -> Self {
        Self::new(space, inside, cubes, &CONV_DYNAMIC)
    }

    fn new(space: Space, inside: Levels, cubes: &[Axis], dynamic: &'static [Axis]) -> Self {
        Self {
            partitioning: Partitioning::new(space, inside.cubes(cubes).batches(&[BATCH]).build()),
            dynamic,
        }
    }

    /// The partitioning a kernel compiles against: the dynamic axes the problem has left to the
    /// launch.
    pub fn kernel(&self) -> Partitioning {
        let space = self.partitioning.space();
        let dynamic: Vec<Axis> = self
            .dynamic
            .iter()
            .copied()
            .filter(|axis| space.contains(*axis))
            .collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The launcher of `grid`, settled over the partitioning's extents on its first launch.
    pub fn launcher(&self, client: &Client, grid: &Grid) -> Launcher {
        grid.launcher(client, || {
            (self.kernel(), self.partitioning.space().clone())
        })
    }

    /// cubek's partitioning, for what the tile engine reads of it: the launch and the binding.
    pub fn as_cubek(&self) -> &Partitioning {
        &self.partitioning
    }
}
