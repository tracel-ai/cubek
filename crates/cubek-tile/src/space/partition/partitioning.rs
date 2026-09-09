//! A [`Space`] and the [`Level`]s that cut it, held as one value.

use cubecl::{prelude::*, unexpanded};

use crate::{Axis, ComputeScope, CubeAxis, Level, Region, RegionExpand, Space, Walk};

/// A space with the levels that partition it: what a kernel's loops are stated over.
///
/// The two are one value because they are never separately true. A `Space` is the axes and
/// their extents; a `Level` cuts *that* space, and its child spaces below. Held apart they are
/// two arguments a call site states in the right order or the wrong one, and nothing says which
/// space a list of levels was cut for — the mismatch compiles, and a level that names an axis
/// the space does not hold, or an edge the extents never asked for, becomes a wrong answer
/// rather than a refusal. Held together, everything read off the pair — the leaf, the overhangs,
/// the grid — is a method, and there is no second space to read it against.
///
/// It is the pair, not a new statement: the levels are the kernel's, outermost first, and
/// nothing here reorders or invents one. What the kernel *does* with a level — where it opens an
/// accumulator, what it stages, which instruction its leaf runs under — stays the kernel's own.
///
/// What a kernel is handed ([`Launcher::partitioning_arg`](crate::Launcher::partitioning_arg)),
/// and what its loops iterate: `for cube in space` deals the first level, `for plane in cube` the
/// next, down to the leaf. The levels ride along comptime; the space's dynamic extents are the
/// runtime half.
#[derive(CubeType, CubeLaunch, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partitioning {
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) levels: Vec<Level>,
}

/// The comptime partitioning of a runtime one, read as the host reads it: what
/// `comptime!(space.clone())` resolves to on the partitioning a kernel is handed.
impl PartitioningExpand {
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Partitioning {
        Partitioning::new(self.space.clone(), self.levels.clone())
    }

    pub fn space(&self) -> Space {
        self.space.clone()
    }

    pub fn levels(&self) -> Vec<Level> {
        self.levels.clone()
    }

    pub fn level(&self, i: usize) -> Level {
        self.levels[i].clone()
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn rank(&self) -> usize {
        self.space.rank()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.space.axis_at(i)
    }

    pub fn extent(&self, axis: Axis) -> usize {
        self.space.extent(axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.space.contains(axis)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.space.position(axis)
    }

    pub fn project(&self, axes: &[Axis]) -> Space {
        self.space.project(axes)
    }

    pub fn axes(&self) -> Vec<Axis> {
        self.space.axes()
    }
}

#[cube]
impl Partitioning {
    /// The regions of the first level: what `for cube in space` iterates.
    pub fn walk(&self) -> Walk {
        Region::root(self).walk()
    }

    /// The regions of `level` over this space, a level of the kernel's own rather than the
    /// partitioning's: a walk the kernel states beside its loops, not one of them.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        Walk::of(&self.space, comptime!(level.clone()), Region::root(self))
    }
}

/// The runtime twin of `for plane in space`, which a kernel's host-side body names but never
/// runs: every loop over a region expands in-kernel.
impl IntoIterator for Partitioning {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl IntoIterator for &Partitioning {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for region in partitioning` iterates its first level.
impl Iterable for PartitioningExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, &self).expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, &self).expand_unroll(scope, body)
    }
}

/// `for cube in &space`: the same, leaving `space` to the body.
impl Iterable for &PartitioningExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, self).expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        RegionExpand::__expand_root(scope, self).expand_unroll(scope, body)
    }
}

impl Partitioning {
    /// `space` cut by `levels`, outermost first.
    pub fn new(space: Space, levels: Vec<Level>) -> Self {
        Self { space, levels }
    }

    /// The space these levels cut.
    pub fn space(&self) -> &Space {
        &self.space
    }

    /// The levels, outermost first — one per loop the kernel writes.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Level `i`, outermost first: what a kernel states its `i`-th loop with.
    pub fn level(&self, i: usize) -> Level {
        self.levels[i].clone()
    }

    /// How many levels there are, which is how deep the nest goes.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// The pair, for a caller that has to hand the halves to something older.
    pub fn into_parts(self) -> (Space, Vec<Level>) {
        (self.space, self.levels)
    }

    /// The space's axes, read through the pair: what a kernel handed a partitioning asks of it.
    pub fn rank(&self) -> usize {
        self.space.rank()
    }

    pub fn axis_at(&self, i: usize) -> Axis {
        self.space.axis_at(i)
    }

    pub fn extent(&self, axis: Axis) -> usize {
        self.space.extent(axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.space.contains(axis)
    }

    pub fn position(&self, axis: Axis) -> usize {
        self.space.position(axis)
    }

    pub fn project(&self, axes: &[Axis]) -> Space {
        self.space.project(axes)
    }

    /// The leaf the levels reach: each level's child of the last, the tile the operands are
    /// cut to at the bottom.
    pub fn leaf(&self) -> Space {
        self.space.leaf(&self.levels)
    }

    /// Whether `axis` overhangs its tiling: some level's edge fails to divide the extent handed
    /// to it, leaving a partial tile that needs masking.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.space.overhangs(&self.levels, axis)
    }

    /// Every axis some tile reaches past the end of, whose accesses are masked.
    pub fn overhanging(&self) -> Vec<Axis> {
        self.space
            .axes()
            .filter(|&axis| self.overhangs(axis))
            .collect()
    }

    /// Instances of `scope` these levels deal the space to: the product, over every level, of
    /// the instance count of each axis riding it, times the instance count of any work a level
    /// distributes as one on it ([`Work`](crate::Work)).
    pub fn instances(&self, scope: ComputeScope) -> u32 {
        let mut total = 1u32;
        let mut space = self.space.clone();
        for level in &self.levels {
            // Work distributed as one rides its scope whole rather than through any one of its
            // axes, so its instance count is the dim's and no axis of it contributes.
            if let Some(work) = level.work()
                && work.scope() == scope
            {
                total *= work.instances() as u32;
            }
            for axis in space.axes() {
                let dist = level.distribution(axis);
                if dist.scope() == Some(scope) {
                    // `count` is `ceil`, so an indivisible axis adds the instance for its
                    // partial tile.
                    total *= dist.coverage().instances(level.count(&space, axis)) as u32;
                }
            }
            space = level.child(&space);
        }
        total
    }

    /// The grid these levels deal to: cube dimension `d` gets the instance count of whichever
    /// axis is `Spatial { Cube(d), .. }`, at any level, else 1.
    pub fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            self.instances(ComputeScope::Cube(CubeAxis::X)),
            self.instances(ComputeScope::Cube(CubeAxis::Y)),
            self.instances(ComputeScope::Cube(CubeAxis::Z)),
        )
    }

    /// Planes one cube holds, read off the levels' plane cuts.
    pub fn planes_per_cube(&self) -> u32 {
        self.instances(ComputeScope::Plane)
    }

    /// Lanes one instance holds, read off the levels' unit cuts. `1` where no level cuts to
    /// units, which is a plan whose leaf the whole plane runs.
    pub fn lanes(&self) -> u32 {
        self.instances(ComputeScope::Unit)
    }

    /// The cube this partitioning asks for at `plane_size`: the plane width by the planes a
    /// cube holds.
    pub fn cube_dim(&self, plane_size: u32) -> CubeDim {
        CubeDim::new_2d(plane_size, self.planes_per_cube())
    }
}
