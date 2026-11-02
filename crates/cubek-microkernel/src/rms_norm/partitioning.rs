use crate::row_store::{ROWS, RowStore, WIDTH};
use cubecl::tile::{
    Axis, Levels, Partitioning, Space,
    space::{ComputeScope, PlaneOrder},
};
use cubek_std::matmul_axes::batch_axis;
use cubek_std::routine::{CubeBoxes, LaunchPlanes, Partitioned};

/// A row: the axis the norm is taken independently along, the rows of the product it feeds.
pub const R: Axis = ROWS;
/// The normalized dim, which every row spans and the weight runs along: the contraction of the
/// product it feeds. A packed store states it as its blocks ([`RowStore::width_axes`]).
pub const D: Axis = WIDTH;
/// The batch the rows come in: each entry's rows lie at their own stride from the last entry's.
pub const B: Axis = batch_axis(0);

/// How a routine of this kind places its microkernel's rows over a problem, handing its boxes out
/// by `P`: a cube for every box of rows of every batch entry, and inside the cube's box the levels
/// the routine and its microkernel state ([`CubeBoxes`]); or a plane of the launch for every row,
/// the launch grouping its planes in cubes as an order says ([`LaunchPlanes`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormPartitioning<P> {
    partitioning: Partitioned<P>,
    /// The box one cube covers.
    cube_box: Space,
    /// How the rows are stored, which decides what the kernel leaves to the launch.
    store: RowStore,
}

impl RmsNormPartitioning<CubeBoxes> {
    /// `space`, every axis of the problem, partitioned with `inside` a cube's box, its rows
    /// stored as `store` says. A cube's box is its rows of one batch entry, all of `D`.
    pub fn new(space: Space, inside: Levels, store: RowStore) -> Self {
        let partitioning = Partitioning::new(space, inside.cubes(&[R]).batches(&[B]).build());
        let cube_box = partitioning.box_of(ComputeScope::Cube);
        Self {
            partitioning: Partitioned::new(partitioning, CubeBoxes),
            cube_box,
            store,
        }
    }
}

impl RmsNormPartitioning<LaunchPlanes> {
    /// `space` partitioned with `inside` a plane's box, every row's box a plane of the launch's,
    /// the planes taking them in `order`: a plain store's alone, since no cube holds the planes.
    /// A cube's box is the rows of its planes, which take consecutive rows.
    pub fn over_planes(space: Space, inside: Levels, order: PlaneOrder) -> Self {
        let planes = order.planes_per_cube();
        let partitioning =
            Partitioning::new(space, inside.launch_planes(&[R, B], order.clone()).build());
        let plane = partitioning.box_of(ComputeScope::Plane);
        let cube_box = Space::new(&[
            (R, plane.extent(R) * planes),
            (D, plane.extent(D)),
            (B, plane.extent(B)),
        ]);
        Self {
            partitioning: Partitioned::new(partitioning, LaunchPlanes { order }),
            cube_box,
            store: RowStore::Plain,
        }
    }
}

impl<P> RmsNormPartitioning<P> {
    /// The box one cube covers.
    pub fn cube_box(&self) -> Space {
        self.cube_box.clone()
    }

    /// The partitioning a kernel compiles against: the batch and the rows of a plain store left
    /// to the launch ([`RowStore::dynamic_axes`]), the dim compiled in, since every owner sizes its
    /// registers by it.
    pub fn kernel(&self) -> Partitioning {
        let dynamic = [&[B], self.store.dynamic_axes()].concat();
        let partitioning = self.as_tile();
        Partitioning::new(
            partitioning.space().clone().with_dynamic(&dynamic),
            partitioning.levels().to_vec(),
        )
    }

    /// The tile engine's partitioning, for the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        self.partitioning.as_tile()
    }
}
