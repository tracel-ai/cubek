use cubecl::tile::{Axis, Levels, Partitioning, Space};

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
/// A channel of the `q|k|v` stream the conv mixes.
pub const CHANNEL: Axis = Axis(8);
/// A tap of the conv's filter.
pub const TAP: Axis = Axis(9);
/// A column of a window store's ring, the conv's past values of every channel.
pub const RING_COLUMN: Axis = Axis(10);
/// A channel of one key head, head local: no operand binds it, only the units level cuts it.
pub const HEAD_CHANNEL: Axis = Axis(11);
/// A position of a prompt that later positions read by tap: no level cuts it.
pub const STREAM_POSITION: Axis = Axis(12);
/// The two rates of a position and value head, the log of its decay at 0 and its strength at 1: no
/// level cuts it.
pub const RATE: Axis = Axis(13);
/// A box of channel lines one cube works on: no operand binds it, only the cube level cuts it.
pub const CHANNEL_BOX: Axis = Axis(14);
/// A line of channels inside a cube's box, box local: no operand binds it, only the units level
/// cuts it.
pub const LINE_IN_BOX: Axis = Axis(15);
/// One of the last `taps - 1` positions of a prompt, which the write back copies into the ring: no
/// operand binds it, only the cube level cuts it.
pub const RING_POSITION: Axis = Axis(16);

/// How the walk places its microkernel's runs over a problem: a cube for every box of value
/// columns of one head of one sequence, and inside the cube's box the levels the routine and its
/// microkernel state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrenceWalkPartitioning {
    partitioning: Partitioning,
}

impl RecurrenceWalkPartitioning {
    /// `space`, every axis of the problem, partitioned with `inside` a cube's box: the boxes of
    /// value columns ride the grid's `x`, the value heads of a key head its `y`, the key heads its
    /// `z`, and the sequences come one a cube.
    pub fn new(space: Space, inside: Levels) -> Self {
        Self {
            partitioning: Partitioning::new(
                space,
                inside
                    .cubes(&[VALUE_FEATURE, HEAD_IN_GROUP, KEY_HEAD])
                    .batches(&[BATCH])
                    .build(),
            ),
        }
    }

    /// The partitioning a kernel compiles against: the sequences, the positions and the store's
    /// slots left to the launch, so a prompt of any length, a decode step of any width and a store
    /// of any size run one kernel; the head dims and the kept states compiled in, since every owner
    /// sizes its registers by the first and the cache's ring fixes the second.
    pub fn kernel(&self) -> Partitioning {
        let space = self.partitioning.space();
        let dynamic: Vec<Axis> = [BATCH, POSITION, STORE_SLOT]
            .into_iter()
            .filter(|axis| space.contains(*axis))
            .collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// cubek's partitioning, for what the tile engine reads of it: the launch and the binding.
    pub fn as_cubek(&self) -> &Partitioning {
        &self.partitioning
    }
}

/// How a conv places its microkernel's work: a cube for every box of channels of every row, and
/// of every position when the space has positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConvPartitioning {
    partitioning: Partitioning,
}

impl CausalConvPartitioning {
    /// `space` partitioned with `inside` a cube's box: the boxes `cube_axis` counts ride the grid's
    /// `x`, or its `y` behind the positions the space has, and rows come one a cube.
    pub fn new(space: Space, inside: Levels, cube_axis: Axis) -> Self {
        let positions = [POSITION, RING_POSITION]
            .into_iter()
            .find(|axis| space.contains(*axis));
        let cubes: Vec<Axis> = positions.into_iter().chain([cube_axis]).collect();
        Self {
            partitioning: Partitioning::new(space, inside.cubes(&cubes).batches(&[BATCH]).build()),
        }
    }

    /// The partitioning a kernel compiles against: rows, positions, slots, ring and prompt left to
    /// the launch.
    pub fn kernel(&self) -> Partitioning {
        let space = self.partitioning.space();
        let dynamic: Vec<Axis> = [BATCH, POSITION, STORE_SLOT, RING_COLUMN, STREAM_POSITION]
            .into_iter()
            .filter(|axis| space.contains(*axis))
            .collect();
        Partitioning::new(
            space.clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// cubek's partitioning, for the launch and the binding.
    pub fn as_cubek(&self) -> &Partitioning {
        &self.partitioning
    }
}
