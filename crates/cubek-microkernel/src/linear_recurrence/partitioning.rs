use cubecl::tile::{Dimension, Levels, Partitioning, Space};

/// A sequence of the walk, or a row of a decode step.
pub const BATCH: Dimension = Dimension(0);
/// A position of the sequence: the dimension the recurrence walks, one step at a time.
pub const POSITION: Dimension = Dimension(1);
/// A key head, which `heads_per_group` value heads read. Mamba2's `ngroups` is the count of these.
pub const KEY_HEAD: Dimension = Dimension(2);
/// A value head's place among the `heads_per_group` heads reading one key head: value head
/// `key_head * heads_per_group + place`.
pub const HEAD_IN_GROUP: Dimension = Dimension(3);
/// A state the walk keeps for the caller to rewind to: the state after each of its last positions,
/// oldest first.
pub const KEPT_STATE: Dimension = Dimension(4);
/// A state row: a key feature, which the queries and keys run along.
pub const KEY_FEATURE: Dimension = Dimension(5);
/// A state column: a value feature, which the values and the output run along.
pub const VALUE_FEATURE: Dimension = Dimension(6);
/// A slot of a store that holds many sequences' states, which a table places each sequence at: no
/// level cuts it.
pub const STORE_SLOT: Dimension = Dimension(7);
/// A channel of the `q|k|v` stream the conv mixes.
pub const CHANNEL: Dimension = Dimension(8);
/// A tap of the conv's filter.
pub const TAP: Dimension = Dimension(9);
/// A column of a window store's ring, the conv's past values of every channel.
pub const RING_COLUMN: Dimension = Dimension(10);
/// A channel of one key head, head local: no operand binds it, only the units level cuts it.
pub const HEAD_CHANNEL: Dimension = Dimension(11);
/// A position of a stream the conv reads by tap: its context columns first, so a stream is
/// `taps - 1` positions longer than the positions it convolves, and no level cuts it.
pub const STREAM_POSITION: Dimension = Dimension(12);

/// How the walk places its microkernel's runs over a problem: a cube for every box of value
/// columns of one head of one sequence, and inside the cube's box the levels the routine and its
/// microkernel state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrenceWalkPartitioning {
    partitioning: Partitioning,
}

impl RecurrenceWalkPartitioning {
    /// `space`, every dimension of the problem, partitioned with `inside` a cube's box: the boxes
    /// of value columns ride the grid's `x`, the value heads of a key head its `y`, the key heads
    /// its `z`, and the sequences come one a cube.
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
        let dynamic: Vec<Dimension> = [BATCH, POSITION, STORE_SLOT]
            .into_iter()
            .filter(|dimension| space.contains(*dimension))
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

/// How the conv places its microkernel's work: a cube for every key head of every row, and of every
/// position when the space has positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkNormedConvPartitioning {
    partitioning: Partitioning,
}

impl QkNormedConvPartitioning {
    /// `space` partitioned with `inside` a cube's box: key heads ride the grid's `x`, or its `y`
    /// behind the positions, and rows come one a cube.
    pub fn new(space: Space, inside: Levels) -> Self {
        let cubes: &[Dimension] = if space.contains(POSITION) {
            &[POSITION, KEY_HEAD]
        } else {
            &[KEY_HEAD]
        };
        Self {
            partitioning: Partitioning::new(space, inside.cubes(cubes).batches(&[BATCH]).build()),
        }
    }

    /// The partitioning a kernel compiles against: rows, positions, slots, ring and stream left to
    /// the launch.
    pub fn kernel(&self) -> Partitioning {
        let space = self.partitioning.space();
        let dynamic: Vec<Dimension> = [BATCH, POSITION, STORE_SLOT, RING_COLUMN, STREAM_POSITION]
            .into_iter()
            .filter(|dimension| space.contains(*dimension))
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
