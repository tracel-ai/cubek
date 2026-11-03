use cubecl::tile::{Axis, Levels, Partitioning, Space};

/// A sequence of the walk, or a row of a decode step.
pub const BATCH: Axis = Axis(0);
/// A position of the sequence: the axis the recurrence walks, one step at a time.
pub const POSITION: Axis = Axis(1);
/// A key head, which `GROUP` value heads read.
pub const KEY_HEAD: Axis = Axis(2);
/// A value head's place among the `group` heads reading one key head: value head
/// `key_head * group + place`.
pub const GROUP: Axis = Axis(3);
/// A state the walk keeps for the caller to rewind to: the state after each of its last positions,
/// oldest first.
pub const KEPT_STATE: Axis = Axis(4);
/// A state row: a key feature, which the queries and keys run along.
pub const KEY_FEATURE: Axis = Axis(5);
/// A state column: a value feature, which the values and the output run along.
pub const VALUE_FEATURE: Axis = Axis(6);

/// How the walk places its microkernel's runs over a problem: a cube for every box of value
/// columns of one head of one sequence, and inside the cube's box the levels the routine and its
/// microkernel state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDeltaWalkPartitioning {
    partitioning: Partitioning,
}

impl GatedDeltaWalkPartitioning {
    /// `space`, every axis of the problem, partitioned with `inside` a cube's box: the boxes of
    /// value columns ride the grid's `x`, the value heads of a key head its `y`, the key heads its
    /// `z`, and the sequences come one a cube.
    pub fn new(space: Space, inside: Levels) -> Self {
        Self {
            partitioning: Partitioning::new(
                space,
                inside
                    .cubes(&[VALUE_FEATURE, GROUP, KEY_HEAD])
                    .batches(&[BATCH])
                    .build(),
            ),
        }
    }

    /// The partitioning a kernel compiles against: the sequences and the positions left to the
    /// launch, so a prompt of any length and a decode step of any width run one kernel; the head
    /// dims and the kept states compiled in, since every owner sizes its registers by the first and
    /// the cache's ring fixes the second.
    pub fn kernel(&self) -> Partitioning {
        Partitioning::new(
            self.partitioning
                .space()
                .clone()
                .with_dynamic(&[BATCH, POSITION]),
            self.partitioning.levels().to_vec(),
        )
    }

    /// cubek's partitioning, for what the tile engine reads of it: the launch and the binding.
    pub fn as_cubek(&self) -> &Partitioning {
        &self.partitioning
    }
}
