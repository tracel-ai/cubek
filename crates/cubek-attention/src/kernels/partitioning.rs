use core::num::NonZeroUsize;
use cubecl::tile::{
    Axis, Level, Levels, Partitioning, Space,
    space::{ComputeScope, Coverage, PlaneOrder, UnitCube},
};
use cubek_microkernel::attention::{B, G, KV, Q, S};
use cubek_std::routine::{
    CubeBoxes, CubeLine, LaunchPlanes, LaunchUnits, Partition, Partitioned, SplitBoxes,
};

/// How a routine of this kind places its microkernel's work items over a problem, handing its
/// boxes out by `P`: cubes for every box of queries and heads of the group, one batch and key head
/// each, and inside the cube's box the levels the routine and its microkernel state
/// ([`CubeBoxes`]).
///
/// A box's rows are its cube's from loading to writing, and its cube walks the whole of `S`
/// alone, but where several cubes share every box ([`SplitBoxes`]): each then walks its own run of
/// `S`, its steps taken in turns with the others', and the cubes take turns at the box's rows to
/// merge them. Over a cache of pages, the walk takes a page at a time, the steps inside it, and
/// cubes sharing a box take the pages in turns. Or every box's steps laid end to end and shared
/// between the cubes ([`CubeLine`]).
///
/// Or, for a routine whose planes share nothing, a plane of the launch for every box of queries
/// and heads, the launch grouping its planes in cubes as an order says ([`LaunchPlanes`]); and for
/// one whose units share nothing, a unit of the launch for every box ([`LaunchUnits`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPartitioning<P> {
    partitioning: Partitioned<P>,
    /// Steps of the walk through `S` a page holds, where the walk takes a page at a time.
    page_steps: Option<NonZeroUsize>,
    /// The box one cube covers, its `S` a cube's run of the walk where several share a box.
    cube_space: Space,
}

/// A [`Partition`] as attention's walk through `S` reads it: which of the cube's walks a stage
/// covers.
pub trait AttentionPartition: Partition {
    /// The box one step of the walk through `S` covers in `partitioning`.
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space;
}

/// The order a launch's cubes take the boxes of queries in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryOrder {
    /// The first box of queries to the first cube launched.
    InOrder,
    /// The last box of queries to the first cube launched: where a later box sees more keys, as
    /// under a causal mask, the heaviest boxes start first and the light ones fill in behind them.
    LastFirst,
}

impl core::fmt::Display for QueryOrder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InOrder => Ok(()),
            Self::LastFirst => write!(f, " queries=last_first"),
        }
    }
}

impl QueryOrder {
    /// `cubes`, their boxes of queries taken in this order, then one batch and key head a cube.
    fn taken_by(self, cubes: Levels) -> Vec<Level> {
        let cubes = match self {
            Self::InOrder => cubes,
            Self::LastFirst => cubes.reversed(Q),
        };
        cubes.batches(&[B, KV]).build()
    }
}

/// `step`, one step of the walk through `S`, its walk taking a page of `page_steps` steps at a
/// time where it has pages.
fn paged(step: Levels, page_steps: Option<NonZeroUsize>) -> Levels {
    match page_steps {
        Some(steps) => step.walk(&[(S, steps.get())]),
        None => step,
    }
}

impl AttentionPartitioning<CubeBoxes> {
    /// `space`, every axis of the problem, partitioned with `step` the levels inside one step of
    /// the walk through `S`, a cube a box, walking the whole of `S`, the boxes of queries taken in
    /// `queries` order. With `page_steps`, the walk takes a page of that many steps at a time.
    pub fn new(
        space: Space,
        step: Levels,
        page_steps: Option<NonZeroUsize>,
        queries: QueryOrder,
    ) -> Self {
        let cubes = paged(step, page_steps).walk_every(&[S]).cubes(&[Q, G]);
        let partitioning =
            Partitioned::new(Partitioning::new(space, queries.taken_by(cubes)), CubeBoxes);
        let cube_space = partitioning.owner_box();
        Self {
            partitioning,
            page_steps,
            cube_space,
        }
    }
}

impl AttentionPartitioning<SplitBoxes> {
    /// [`new`](AttentionPartitioning::new) with the cubes of `split` sharing every box, each taking
    /// a run of the walk (the tile DSL's [`Levels::across`]).
    ///
    /// The count is the one asked for, whatever the keys: it is part of what a kernel compiles,
    /// and the keys are the launch's, any length a cache holds. A buffer of fewer steps than the
    /// cubes leaves the cubes past them nothing to walk, as a short attended prefix does: they
    /// leave an empty partial the merge reads. The cubes take the walk's steps, or pages, in turns
    /// ([`Levels::interleaved`]), so the attended keys, a prefix of the buffers', are spread over
    /// every cube with a step or page of them: contiguous runs would leave the cubes past the
    /// prefix with nothing to walk. Where the count does not divide them, the trailing cubes take
    /// one fewer, or none.
    pub fn split(
        space: Space,
        step: Levels,
        page_steps: Option<NonZeroUsize>,
        split: SplitBoxes,
        queries: QueryOrder,
    ) -> Self {
        let cubes = paged(step, page_steps)
            .cubes(&[Q, G, S])
            .across(S, split.cubes.get())
            .interleaved(S);
        let keys = space.extent(S);
        let partitioning =
            Partitioned::new(Partitioning::new(space, queries.taken_by(cubes)), split);
        // The cube level's own tile is one step of `S`, or one page, distributed in runs: the box
        // states the run, as long as the longest cube's.
        let tile = partitioning.owner_box();
        let run = split.run(keys, tile.extent(S));
        let cube_space = Space::new(
            &tile
                .extents()
                .into_iter()
                .map(|(axis, extent)| (axis, if axis == S { run } else { extent }))
                .collect::<Vec<_>>(),
        );
        Self {
            partitioning,
            page_steps,
            cube_space,
        }
    }

    /// Cubes sharing every box of rows, each on its own run of `S`.
    pub fn cubes_along_keys(&self) -> NonZeroUsize {
        self.partitioning.partition().cubes
    }

    /// Cubes one batch and key head of the problem spawns: the cubes sharing each box, for every
    /// box of queries and heads.
    pub fn cubes(&self) -> usize {
        self.boxes() * self.cubes_along_keys().get()
    }
}

impl AttentionPartitioning<CubeLine> {
    /// `space`, every axis of the problem, partitioned with `step` the levels inside one step of
    /// the walk through `S`, every box of queries and heads holding the whole of `S`, and the steps
    /// of each batch and key head's boxes laid end to end and shared by the cubes of `line`, each
    /// taking a run that may start inside one box and end inside the next (the tile DSL's
    /// [`Levels::shared_by`]): every cube walks as many steps as any other, however the boxes
    /// divide them.
    pub fn shared(space: Space, step: Levels, line: CubeLine) -> Self {
        let levels = step
            .walk_every(&[S])
            .cubes(&[Q, G])
            .shared_by(line.cubes.get())
            .batches(&[B, KV])
            .build();
        let partitioning = Partitioned::new(Partitioning::new(space, levels), line);
        let cube_space = partitioning.owner_box();
        Self {
            partitioning,
            page_steps: None,
            cube_space,
        }
    }

    /// Cubes one batch and key head of the problem spawns: the cubes sharing the line of its
    /// boxes' steps.
    pub fn cubes(&self) -> usize {
        self.partitioning.partition().cubes.get()
    }
}

impl AttentionPartitioning<LaunchUnits> {
    /// `space`, every axis of the problem, partitioned with `inside` a unit's box, every box of
    /// queries and heads a unit of the launch's, one batch and key head each, the units taking
    /// them in row order, the queries fastest, grouped in cubes of `cube`. A cube's box is the
    /// units it holds, laid along the queries.
    pub fn over_units(space: Space, inside: Levels, cube: UnitCube) -> Self {
        let units = cube.units();
        let partitioning =
            Partitioned::launch_units(space, inside, &[Q, G, B, KV], LaunchUnits { cube });
        Self::spanning_launch(partitioning, units)
    }
}

impl AttentionPartitioning<LaunchPlanes> {
    /// `space`, every axis of the problem, partitioned with `inside` a plane's box, every box of
    /// queries and heads a plane of the launch's, one batch and key head each, the planes taking
    /// them in `order`. A cube's box is the planes it holds, laid along the queries.
    pub fn over_planes(space: Space, inside: Levels, order: PlaneOrder) -> Self {
        let planes = order.planes_per_cube();
        let partitioning =
            Partitioned::launch_planes(space, inside, &[Q, G, B, KV], LaunchPlanes { order });
        Self::spanning_launch(partitioning, planes)
    }
}

impl<P: Partition> AttentionPartitioning<P> {
    /// `partitioning`, whose outermost level hands every box of queries and heads to an owner of
    /// the launch, `per_cube` of them a cube. A cube's box is its owners' laid along the queries.
    fn spanning_launch(partitioning: Partitioned<P>, per_cube: usize) -> Self {
        let owner = partitioning.owner_box();
        let cube_space = Space::new(
            &owner
                .extents()
                .into_iter()
                .map(|(axis, extent)| match axis == Q {
                    true => (axis, extent * per_cube),
                    false => (axis, extent),
                })
                .collect::<Vec<_>>(),
        );
        Self {
            partitioning,
            page_steps: None,
            cube_space,
        }
    }

    /// The tile of the walk through `S` inside a cube past `outer` such walks.
    fn walk_inside_cube(&self, outer: usize) -> Space {
        // Where the planes or units span the launch, the levels inside the outermost cut an
        // owner's box.
        let mut enclosing = match P::OWNER {
            ComputeScope::Cube => self.cube_space(),
            _ => self.partitioning.owner_box(),
        };
        let mut walks = self.as_tile().levels()[1..].iter().filter_map(|level| {
            let tile = level.child(&enclosing);
            enclosing = tile.clone();
            (level.coverage() == Coverage::Walk && level.tile(S).is_some()).then_some(tile)
        });
        walks
            .nth(outer)
            .expect("the partitioning walks no S inside its cubes")
    }
}

impl AttentionPartition for CubeBoxes {
    /// A cube walks the pages, then the steps inside each; or the steps of its own tile.
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space {
        match partitioning.page_steps {
            Some(_) => partitioning.walk_inside_cube(1),
            None => partitioning.walk_inside_cube(0),
        }
    }
}

impl AttentionPartition for SplitBoxes {
    /// The cube level steps `S` itself, a run each, so its tile is one step; or, where cubes take
    /// pages in turns, the page is walked in steps.
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space {
        match partitioning.page_steps {
            Some(_) => partitioning.walk_inside_cube(0),
            None => partitioning.partitioning.owner_box(),
        }
    }
}

impl AttentionPartition for CubeLine {
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space {
        partitioning.walk_inside_cube(0)
    }
}

impl AttentionPartition for LaunchPlanes {
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space {
        partitioning.walk_inside_cube(0)
    }
}

impl AttentionPartition for LaunchUnits {
    fn stage(partitioning: &AttentionPartitioning<Self>) -> Space {
        partitioning.walk_inside_cube(0)
    }
}

impl<P: AttentionPartition> AttentionPartitioning<P> {
    /// The box one step of the walk through `S` covers: what a staged routine stages at once.
    ///
    /// # Panics
    ///
    /// A partitioning whose cubes walk no `S`, alone or in runs.
    pub fn stage(&self) -> Space {
        P::stage(self)
    }
}

impl<P> AttentionPartitioning<P> {
    /// The box one cube covers: its queries of its heads, all of `D` and `DV`, and all of `S`, or
    /// its run of it where several cubes share the box.
    pub fn cube_space(&self) -> Space {
        self.cube_space.clone()
    }

    /// The innermost box: what one instruction, or one unit's block, covers.
    pub fn leaf(&self) -> Space {
        self.as_tile().leaf()
    }

    /// Whether the tiles overhang `axis`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.as_tile().overhangs(axis)
    }

    /// The problem's space: every axis at its extent, or dynamic where a launch states it.
    pub fn space(&self) -> &Space {
        self.as_tile().space()
    }

    /// Boxes of queries and heads one batch and key head of the problem holds.
    pub fn boxes(&self) -> usize {
        let (space, cube) = (self.space(), self.cube_space());
        let boxes = |axis| space.extent(axis).div_ceil(cube.extent(axis));
        boxes(Q) * boxes(G)
    }

    /// The partitioning a kernel compiles against: every extent a launch varies — the batches, the
    /// key heads, the queries and the keys — left to the launch, and the group and the dims a row
    /// spans compiled in, which a row's registers and fragments are sized by.
    pub fn kernel(&self) -> Partitioning {
        Partitioning::new(
            self.space().clone().with_dynamic(&[B, KV, Q, S]),
            self.as_tile().levels().to_vec(),
        )
    }

    /// The tile DSL's partitioning, for what it reads of it: the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        self.partitioning.as_tile()
    }
}
