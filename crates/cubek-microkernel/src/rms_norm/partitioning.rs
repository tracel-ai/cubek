use crate::row_store::{ROWS, RowStore, WIDTH};
use cubecl::tile::{
    Axis, Levels, Partitioning, Space,
    space::{ComputeScope, PlaneOrder},
};
use cubek_std::matmul_axes::batch_axis;

/// A row: the axis the norm is taken independently along, the rows of the product it feeds.
pub const R: Axis = ROWS;
/// The normalized dim, which every row spans and the weight runs along: the contraction of the
/// product it feeds. A packed store states it as its blocks ([`RowStore::width_axes`]).
pub const D: Axis = WIDTH;
/// The batch the rows come in: each entry's rows lie at their own stride from the last entry's.
pub const B: Axis = batch_axis(0);

/// How a routine of this kind places its microkernel's rows over a problem: a cube for every box
/// of rows of every batch entry, and inside the cube's box the levels the routine and its
/// microkernel state; or, for a routine whose planes share nothing, a plane of the launch for every
/// row, the launch grouping its planes in cubes as an order says
/// ([`over_planes`](Self::over_planes)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormPartitioning {
    partitioning: Partitioning,
    /// How the rows are stored, which decides what the kernel leaves to the launch.
    store: RowStore,
}

impl RmsNormPartitioning {
    /// `space`, every axis of the problem, partitioned with `inside` a cube's box, its rows
    /// stored as `store` says.
    pub fn new(space: Space, inside: Levels, store: RowStore) -> Self {
        Self {
            partitioning: Partitioning::new(space, inside.cubes(&[R]).batches(&[B]).build()),
            store,
        }
    }

    /// `space` partitioned with `inside` a plane's box, every row's box a plane of the launch's,
    /// the planes taking them in `order`: a plain store's alone, since no cube holds the planes.
    pub fn over_planes(space: Space, inside: Levels, order: PlaneOrder) -> Self {
        Self {
            partitioning: Partitioning::new(space, inside.launch_planes(&[R, B], order).build()),
            store: RowStore::Plain,
        }
    }

    /// The box one cube covers: its rows of one batch entry, all of `D`. Where the planes span the
    /// launch, the rows of a cube's planes, which take consecutive rows.
    pub fn cube_box(&self) -> Space {
        let outer = &self.partitioning.levels()[0];
        match outer.plane_order() {
            None => self.partitioning.box_of(ComputeScope::Cube),
            Some(order) => {
                let plane = self.partitioning.box_of(ComputeScope::Plane);
                Space::new(&[
                    (R, plane.extent(R) * order.planes_per_cube()),
                    (D, plane.extent(D)),
                    (B, plane.extent(B)),
                ])
            }
        }
    }

    /// The partitioning a kernel compiles against: the batch and the rows of a plain store left
    /// to the launch ([`RowStore::dynamic_axes`]), the dim compiled in, since every owner sizes its
    /// registers by it.
    pub fn kernel(&self) -> Partitioning {
        let dynamic = [&[B], self.store.dynamic_axes()].concat();
        Partitioning::new(
            self.partitioning.space().clone().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The tile engine's partitioning, for the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        &self.partitioning
    }
}
