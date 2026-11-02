use core::fmt::Debug;
use core::num::NonZeroUsize;
use cubecl::tile::space::{ComputeScope, PlaneOrder, UnitCube};
use cubecl::tile::{Axis, Levels, Partitioning, Space};

/// How a launch hands out the boxes of its problem: the scope that owns a box, and the cubes that
/// share one. Every launch of every routine hands its boxes out one way, so a routine names its
/// way as a type, and a family's partitioning is generic over it.
///
/// The way holds what it is of the decision: how many cubes share a box, the order the launch's
/// planes are grouped in. The levels the routine builds from it, inside and around, are the tile
/// DSL's ([`Partitioning`]).
pub trait Partition: Debug + Clone + PartialEq + Eq {
    /// The scope that owns a box: a cube, or a plane or unit of the launch, which a cube only
    /// groups.
    const OWNER: ComputeScope;
}

/// Every cube owns a box, and walks the whole of it alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeBoxes;

/// Every box is shared by `cubes` cubes, each walking its own run of the box's walk; their
/// partial results are merged after (the tile DSL's [`Levels::across`](cubecl::tile::Levels::across)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplitBoxes {
    pub cubes: NonZeroUsize,
}

/// The steps of every box's walk laid end to end and cut in equal runs between `cubes` cubes, a
/// run free to start inside one box and end inside the next, so every cube walks as many steps as
/// any other (the tile DSL's [`Levels::shared_by`](cubecl::tile::Levels::shared_by)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeLine {
    pub cubes: NonZeroUsize,
}

/// Every plane of the launch owns a box, the planes taking them in `order`; a cube only groups
/// planes, sharing nothing between them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LaunchPlanes {
    pub order: PlaneOrder,
}

/// Every unit of the launch owns a box; a cube only groups units, laid as `cube` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchUnits {
    pub cube: UnitCube,
}

impl Partition for CubeBoxes {
    const OWNER: ComputeScope = ComputeScope::Cube;
}

impl Partition for SplitBoxes {
    const OWNER: ComputeScope = ComputeScope::Cube;
}

impl Partition for CubeLine {
    const OWNER: ComputeScope = ComputeScope::Cube;
}

impl Partition for LaunchPlanes {
    const OWNER: ComputeScope = ComputeScope::Plane;
}

impl Partition for LaunchUnits {
    const OWNER: ComputeScope = ComputeScope::Unit;
}

impl SplitBoxes {
    /// The run of a box's walk each cube takes: the walk's `steps`, each `step` long, cut between
    /// the cubes and rounded up, as long as the longest cube's. A walk of fewer steps than cubes
    /// leaves the trailing cubes an empty run.
    pub fn run(&self, walk: usize, step: usize) -> usize {
        walk.div_ceil(step).max(1).div_ceil(self.cubes.get()) * step
    }
}

/// The tile DSL's partitioning of a problem, with the [`Partition`] its outer levels were built
/// from: what a family with nothing of its own to add places a launch by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partitioned<P> {
    partitioning: Partitioning,
    partition: P,
}

impl<P> Partitioned<P> {
    /// `partitioning`, its outer levels built from `partition`.
    pub fn new(partitioning: Partitioning, partition: P) -> Self {
        Self {
            partitioning,
            partition,
        }
    }

    /// The way its boxes are handed out.
    pub fn partition(&self) -> &P {
        &self.partition
    }

    /// The tile DSL's partitioning, for the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        &self.partitioning
    }
}

impl Partitioned<CubeBoxes> {
    /// `space` partitioned with `inside` a cube's box, the cubes cutting `axes`, one entry of each
    /// of `batches` a cube.
    pub fn cubes(space: Space, inside: Levels, axes: &[Axis], batches: &[Axis]) -> Self {
        let levels = inside.cubes(axes).batches(batches).build();
        Self::new(Partitioning::new(space, levels), CubeBoxes)
    }
}

impl Partitioned<LaunchPlanes> {
    /// `space` partitioned with `inside` a plane's box, every box `axes` cut a plane of the
    /// launch's, the planes taking them as `partition` orders them.
    pub fn launch_planes(
        space: Space,
        inside: Levels,
        axes: &[Axis],
        partition: LaunchPlanes,
    ) -> Self {
        let levels = inside.launch_planes(axes, partition.order.clone()).build();
        Self::new(Partitioning::new(space, levels), partition)
    }
}

impl Partitioned<LaunchUnits> {
    /// `space` partitioned with `inside` a unit's box, every box `axes` cut a unit of the launch's,
    /// grouped in cubes as `partition` lays them.
    pub fn launch_units(
        space: Space,
        inside: Levels,
        axes: &[Axis],
        partition: LaunchUnits,
    ) -> Self {
        let levels = inside.launch_units(axes, partition.cube).build();
        Self::new(Partitioning::new(space, levels), partition)
    }
}

impl<P: Partition> Partitioned<P> {
    /// The box one owner covers ([`Partition::OWNER`]).
    pub fn owner_box(&self) -> Space {
        self.partitioning.box_of(P::OWNER)
    }
}
