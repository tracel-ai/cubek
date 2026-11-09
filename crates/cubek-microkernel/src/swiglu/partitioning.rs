use crate::row_store::{ROWS, RowStore, WIDTH};
use cubecl::tile::{Axis, Levels, Partitioning, Space, space::ComputeScope};
use cubek_std::matmul_axes::batch_axis;

/// A row: the axis the halves are paired along, one row of the gate beside one of the up, and the
/// rows of the product the output feeds.
pub const R: Axis = ROWS;
/// The half of a fused row, two wide: the gate, then the up. The output does not run along it.
/// The label of the matmul axes' first batch, which nothing in this family's space batches over:
/// a label of cubek's, so no axis it adds to the space a packed matrix binds over can take it.
pub const H: Axis = batch_axis(0);
/// The width of a half, which the output takes: the contraction of the product it feeds. A packed
/// store states it as its blocks ([`RowStore::width_axes`]).
pub const D: Axis = WIDTH;

/// How a routine of this kind places its microkernel's runs over a problem: a cube for every box
/// of rows and of a run of their width, both halves of it, and inside the cube's box the levels
/// the routine and its microkernel state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwigluPartitioning {
    partitioning: Partitioning,
    /// How the output's rows are stored, which decides what the kernel leaves to the launch.
    store: RowStore,
}

impl SwigluPartitioning {
    /// `space`, every axis of the problem, partitioned with `inside` a cube's box, the rows stored
    /// as `store` says: the boxes of rows ride the grid's `x`, the most cubes a device spawns along
    /// any axis, and the runs of a row its `y`.
    pub fn new(space: Space, inside: Levels, store: RowStore) -> Self {
        Self {
            partitioning: Partitioning::new(space, inside.cubes(&[R, store.cut()]).build()),
            store,
        }
    }

    /// The box one cube covers: its rows, and the run of their width it holds.
    pub fn cube_box(&self) -> Space {
        self.partitioning.box_of(ComputeScope::Cube)
    }

    /// The partitioning a kernel compiles against: the rows of a plain store left to the launch
    /// ([`RowStore::dynamic_axes`]), the width compiled in, since it is a model's `d_ff` and the
    /// autotune key holds it exactly.
    pub fn kernel(&self) -> Partitioning {
        Partitioning::new(
            self.partitioning
                .space()
                .clone()
                .with_dynamic(self.store.dynamic_axes()),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The tile engine's partitioning, for the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        &self.partitioning
    }
}
