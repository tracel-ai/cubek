//! A [`Space`] and the [`Level`]s that cut it, held as one value.

use cubecl::{prelude::*, unexpanded};

use crate::{Axis, Count, CubeAxis, Level, LevelTable, Region, RegionExpand, Space, Takers, Walk};

/// A space with the levels that partition it: what a kernel's loops are stated over.
///
/// One value because they are never separately true: a `Level` cuts *that* `Space`. Held apart,
/// nothing says which space the levels were cut for: a level naming an axis the space lacks is a
/// wrong answer, not a refusal. Held together, every read is a method; there is no second space.
///
/// It is the pair, not a new statement: the levels are the kernel's, outermost first, and
/// nothing here reorders or invents one. What the kernel *does* with a level — where it opens an
/// accumulator, what it stages, which instruction its leaf runs under — stays the kernel's own.
///
/// What a kernel is handed ([`Launcher::partitioning_arg`](crate::Launcher::partitioning_arg)),
/// and what its loops iterate: `for cube in space` deals the first level, `for plane in cube` the
/// next, down to the leaf. Levels are comptime; the space's dynamic extents are the runtime half.
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
}

/// The comptime reads of a partitioning a kernel is handed: its levels, and its space.
impl PartitioningExpand {
    /// The levels, outermost first: one per loop the kernel writes.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// How many levels there are, which is how deep the nest goes.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
}

impl Partitioning {
    /// The levels, outermost first: one per loop the kernel writes.
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

    /// This partitioning in kernel form: every extent [`Dynamic`](crate::Extent::Dynamic), the
    /// levels unchanged, so one compiled kernel serves every shape a launch stamps on.
    pub fn all_dynamic(self) -> Self {
        Partitioning {
            space: self.space.all_dynamic(),
            levels: self.levels,
        }
    }

    /// This partitioning with a name for each of its axes, which is what prints a table worth
    /// reading: an [`Axis`] is a client-assigned index, so only the client can say what it
    /// stands for. An axis the labels do not name prints that index.
    pub fn table<'a>(&'a self, labels: &'a [(Axis, &'a str)]) -> LevelTable<'a> {
        LevelTable::new(self, labels)
    }

    /// The leaf the levels reach: each level's child of the last, the tile the operands are
    /// cut to at the bottom.
    pub fn leaf(&self) -> Space {
        self.space.leaf(&self.levels)
    }

    /// Whether `axis` overhangs its tiling: some level's edge fails to divide the extent handed
    /// to it, leaving a partial tile that needs masking.
    pub fn overhangs(&self, axis: Axis) -> bool {
        assert!(
            !self.space.is_dynamic(axis),
            "Partitioning::overhangs: axis {axis:?} is Dynamic; ask the concrete space"
        );
        let mut space = self.space.clone();
        for level in &self.levels {
            if level.overhangs(&space, axis) {
                return true;
            }
            space = level.child(&space);
        }
        false
    }

    /// Every axis some tile reaches past the end of, whose accesses are masked.
    pub fn overhanging(&self) -> Vec<Axis> {
        self.space
            .axes()
            .filter(|&axis| self.overhangs(axis))
            .collect()
    }

    /// The one level whose tiles `takers` take, if any. A partitioning has at most one lanes,
    /// planes and cubes level; its walks may be several ([`walks`](Self::walks)).
    pub fn level_of(&self, takers: Takers) -> Option<&Level> {
        assert!(
            takers != Takers::Walk,
            "Partitioning::level_of: a partitioning may walk several levels; ask `walks`"
        );
        let mut found = self.levels.iter().filter(|level| level.takers() == takers);
        let level = found.next();
        assert!(
            found.next().is_none(),
            "Partitioning::level_of: two levels are taken by {takers:?}"
        );
        level
    }

    /// The levels one instance walks, outermost first.
    pub fn walks(&self) -> impl Iterator<Item = &Level> + '_ {
        self.levels
            .iter()
            .filter(|level| level.takers() == Takers::Walk)
    }

    /// Instances `takers` deal the space to: the product, over every level they take, of the
    /// instance count of each axis it deals, or of the workers sharing its grid as one. Cubes
    /// count per grid dimension ([`cube_instances`](Self::cube_instances)).
    pub(crate) fn instances(&self, takers: Takers) -> u32 {
        match takers {
            Takers::Cubes => [CubeAxis::X, CubeAxis::Y, CubeAxis::Z]
                .into_iter()
                .map(|dim| self.cube_instances(dim))
                .product(),
            Takers::Planes | Takers::Lanes => {
                self.count_instances(|level| level.takers() == takers, |_, _| true)
            }
            Takers::Walk => 1,
        }
    }

    /// Cubes on grid dimension `dim`: the instance count of whichever axis rides it, at any level,
    /// else one. A grid shared as one index rides `X` whole.
    fn cube_instances(&self, dim: CubeAxis) -> u32 {
        self.count_instances(
            |level| {
                level.takers() == Takers::Cubes
                    && (level.shared_by().is_none() || dim == CubeAxis::X)
            },
            |level, axis| level.cube_axis(axis) == Some(dim),
        )
    }

    /// The product over the `levels` selected of the workers sharing the grid as one, or of the
    /// tiles each `axes` selected is dealt in. Each level's count is read against the space its
    /// parents hand it; `tiles` is `ceil`, so an indivisible axis adds the instance for its
    /// partial tile.
    fn count_instances(
        &self,
        levels: impl Fn(&Level) -> bool,
        axes: impl Fn(&Level, Axis) -> bool,
    ) -> u32 {
        let mut total = 1u32;
        let mut space = self.space.clone();
        for level in &self.levels {
            if levels(level) {
                match level.shared_by() {
                    Some(workers) => total *= workers as u32,
                    None => {
                        for axis in space.axes() {
                            if level.deals(axis) && axes(level, axis) {
                                total *= match level.count(axis) {
                                    Some(Count::AllAcross(workers)) => workers,
                                    // Tiles dealt to as many lanes as the launch runs ask for
                                    // none of their own.
                                    Some(Count::Dealt(_)) => 1,
                                    _ => level.tiles(&space, axis),
                                } as u32;
                            }
                        }
                    }
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
            self.cube_instances(CubeAxis::X),
            self.cube_instances(CubeAxis::Y),
            self.cube_instances(CubeAxis::Z),
        )
    }

    /// Planes one cube holds: the levels' plane cuts, plus any that only fill.
    pub fn planes_per_cube(&self) -> u32 {
        self.instances(Takers::Planes) + self.fillers()
    }

    /// Planes the cube holds that fill a walk's stages and take no tile
    /// ([`Level::filled_by`]).
    ///
    /// Added to the instance count rather than multiplied into it, and read by nothing else: a
    /// filling plane is a disjoint set of planes, not a cut of the ones that compute, and no
    /// level decode has anything to read off it.
    pub fn fillers(&self) -> u32 {
        self.levels.iter().map(|level| level.fillers() as u32).sum()
    }

    /// Lanes one instance holds, read off the levels' unit cuts. `1` where no level cuts to
    /// units, which is a plan whose leaf the whole plane runs, or where the lanes take their
    /// tiles in turns ([`Count::Dealt`]), however many the launch runs.
    pub fn lanes(&self) -> u32 {
        self.instances(Takers::Lanes)
    }

    /// The cube this partitioning asks for at `plane_size`: the plane width by the planes a
    /// cube holds.
    pub fn cube_dim(&self, plane_size: u32) -> CubeDim {
        CubeDim::new_2d(plane_size, self.planes_per_cube())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Levels;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A cube grid over `M`/`N`, a `K` walk, then one partition per plane: the staged shape,
    /// with `fillers` planes filling the walk's stages.
    fn staged(fillers: usize) -> Partitioning {
        Partitioning::new(
            Space::new(&[(M, 256), (N, 256), (K, 512)]),
            Levels::leaf(&[(M, 64), (N, 64), (K, 64)])
                .planes(&[(M, 2), (N, 2)])
                .walk_every(&[K])
                .filled_by(fillers)
                .cubes(&[M, N])
                .build(),
        )
    }

    #[test]
    fn a_walk_with_no_fillers_is_the_partitioning_it_always_was() {
        let plain = staged(0);
        assert_eq!(plain.fillers(), 0);
        assert_eq!(plain.planes_per_cube(), 4);
        assert_eq!(plain.cube_dim(32), CubeDim::new_2d(32, 4));
    }

    /// The cube is wider by the count and nothing else moves: the instances a level deals, and
    /// so every position decoded below it, are the ones they were.
    #[test]
    fn a_filling_plane_widens_the_cube_and_nothing_else() {
        let plain = staged(0);
        let filled = staged(1);

        assert_eq!(filled.planes_per_cube(), plain.planes_per_cube() + 1);
        assert_eq!(filled.cube_dim(32), CubeDim::new_2d(32, 5));

        assert_eq!(
            filled.instances(Takers::Planes),
            plain.instances(Takers::Planes)
        );
        assert_eq!(filled.lanes(), plain.lanes());
        for dim in [CubeAxis::X, CubeAxis::Y, CubeAxis::Z] {
            assert_eq!(filled.cube_instances(dim), plain.cube_instances(dim));
        }
    }

    #[test]
    fn the_fillers_of_every_level_are_counted() {
        assert_eq!(staged(3).fillers(), 3);
    }

    /// Only a walk's regions are staged, so only a walk can say who fills them.
    #[test]
    #[should_panic(expected = "only a walk's regions are staged")]
    fn a_level_that_deals_its_tiles_cannot_be_filled_by_anyone() {
        Levels::leaf(&[(M, 64)])
            .planes(&[(M, 2)])
            .filled_by(1)
            .build();
    }
}
