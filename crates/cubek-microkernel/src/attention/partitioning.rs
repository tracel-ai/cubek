use crate::attention::{B, G, KV, Q, S};
use core::num::NonZeroUsize;
use cubecl::tile::{
    Axis, Levels, Partitioning, Space,
    space::{ComputeScope, Coverage, PlaneOrder, UnitCube},
};

/// How a routine of this kind places its microkernel's work items over a problem: cubes for
/// every box of queries and heads of the group, one batch and key head each, and inside the
/// cube's box the levels the routine and its microkernel state.
///
/// A box's rows are its cube's from loading to writing, and its cube walks the whole of `S`
/// alone, but where several cubes share every box: each then walks its own run of `S`, its steps
/// taken in turns with the others', and the cubes take turns at the box's rows to merge them.
/// Over a cache of pages, the walk takes a page at a time, the steps inside it, and cubes sharing
/// a box take the pages in turns.
///
/// Or, for a routine whose planes share nothing, a plane of the launch for every box of queries
/// and heads, the launch grouping its planes in cubes as an order says
/// ([`over_planes`](Self::over_planes)); and for one whose units share nothing, a unit of the
/// launch for every box ([`over_units`](Self::over_units)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPartitioning {
    partitioning: Partitioning,
    cubes_along_keys: NonZeroUsize,
    /// Steps of the walk through `S` a page holds, where the walk takes a page at a time.
    page_steps: Option<NonZeroUsize>,
    /// The box one cube covers, its `S` a cube's run of the walk where several share a box.
    cube_box: Space,
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

impl AttentionPartitioning {
    /// One cube a box of rows: the count every routine but the hypercube's distributes.
    pub const ONE_CUBE_A_BOX: NonZeroUsize = NonZeroUsize::MIN;

    /// `space`, every axis of the problem, partitioned with `step` the levels inside one step of
    /// the walk through `S`, and `cubes_along_keys` cubes sharing every box
    /// ([`ONE_CUBE_A_BOX`](Self::ONE_CUBE_A_BOX) for a box to itself), each taking a run of the walk
    /// (cubek's [`Levels::across`]), the boxes of queries taken in `queries` order. With
    /// `page_steps`, the walk takes a page of that many steps at a time.
    ///
    /// The count is the one asked for, whatever the keys: it is part of what a kernel compiles,
    /// and the keys are the launch's. The cubes take the walk's steps, or pages, in turns
    /// ([`Levels::interleaved`]), so the attended keys, a prefix of the buffers', are spread over
    /// every cube with a step or page of them: contiguous runs would leave the cubes past the
    /// prefix with nothing to walk. Where the count does not divide them, the trailing cubes take
    /// one fewer, or none.
    pub fn new(
        space: Space,
        step: Levels,
        page_steps: Option<NonZeroUsize>,
        cubes_along_keys: NonZeroUsize,
        queries: QueryOrder,
    ) -> Self {
        let walked = match page_steps {
            Some(steps) => step.walk(&[(S, steps.get())]),
            None => step,
        };
        let cubes = match cubes_along_keys.get() {
            1 => walked.walk_every(&[S]).cubes(&[Q, G]),
            cubes => walked.cubes(&[Q, G, S]).across(S, cubes).interleaved(S),
        };
        let cubes = match queries {
            QueryOrder::InOrder => cubes,
            QueryOrder::LastFirst => cubes.reversed(Q),
        };
        let levels = cubes.batches(&[B, KV]).build();
        let keys = space.extent(S);
        let partitioning = Partitioning::new(space, levels);
        // Where cubes share a box, the cube level's own tile is one step of `S`, or one page,
        // distributed in runs: the box states the run, as long as the longest cube's.
        let tile = partitioning.box_of(ComputeScope::Cube);
        let run = match cubes_along_keys.get() {
            1 => tile.extent(S),
            cubes => {
                let step = tile.extent(S);
                keys.div_ceil(step).max(1).div_ceil(cubes) * step
            }
        };
        let cube_box = Space::new(
            &tile
                .extents()
                .into_iter()
                .map(|(axis, extent)| (axis, if axis == S { run } else { extent }))
                .collect::<Vec<_>>(),
        );
        Self {
            partitioning,
            cubes_along_keys,
            page_steps,
            cube_box,
        }
    }

    /// `space`, every axis of the problem, partitioned with `inside` a unit's box, every box of
    /// queries and heads a unit of the launch's, one batch and key head each, the units taking
    /// them in row order, the queries fastest, grouped in cubes of `cube`. A cube's box is the
    /// units it holds, laid along the queries.
    pub fn over_units(space: Space, inside: Levels, cube: UnitCube) -> Self {
        let units = cube.units();
        let levels = inside.launch_units(&[Q, G, B, KV], cube);
        Self::spanning_launch(space, levels, ComputeScope::Unit, units)
    }

    /// `space`, every axis of the problem, partitioned with `inside` a plane's box, every box of
    /// queries and heads a plane of the launch's, one batch and key head each, the planes taking
    /// them in `order`. A cube's box is the planes it holds, laid along the queries.
    pub fn over_planes(space: Space, inside: Levels, order: PlaneOrder) -> Self {
        let planes = order.planes_per_cube();
        let levels = inside.launch_planes(&[Q, G, B, KV], order);
        Self::spanning_launch(space, levels, ComputeScope::Plane, planes)
    }

    /// `space` partitioned by `levels`, whose outermost hands every box of queries and heads to an
    /// instance of `scope` of the launch, `per_cube` of them a cube. A cube's box is its
    /// instances' laid along the queries.
    fn spanning_launch(space: Space, levels: Levels, scope: ComputeScope, per_cube: usize) -> Self {
        let partitioning = Partitioning::new(space, levels.build());
        let instance = partitioning.box_of(scope);
        let cube_box = Space::new(
            &instance
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
            cubes_along_keys: Self::ONE_CUBE_A_BOX,
            page_steps: None,
            cube_box,
        }
    }

    /// The box one cube covers: its queries of its heads, all of `D` and `DV`, and all of `S`, or
    /// its run of it where several cubes share the box.
    pub fn cube_box(&self) -> Space {
        self.cube_box.clone()
    }

    /// Cubes sharing every box of rows, each on its own run of `S`.
    pub fn cubes_along_keys(&self) -> NonZeroUsize {
        self.cubes_along_keys
    }

    /// The box one step of the walk through `S` covers: what a staged routine stages at once.
    ///
    /// # Panics
    ///
    /// A partitioning whose cubes walk no `S`, alone or in runs.
    pub fn stage(&self) -> Space {
        match (self.cubes_along_keys.get() > 1, self.page_steps) {
            // The cube level steps `S` itself, a run each: its tile is one step.
            (true, None) => self.partitioning.box_of(ComputeScope::Cube),
            // A cube walks the pages, then the steps inside each.
            (false, Some(_)) => self.walk_inside_cube(1),
            // The cube's own tile, a page where cubes take pages in turns, is walked in steps.
            (_, _) => self.walk_inside_cube(0),
        }
    }

    /// The tile of the walk through `S` inside a cube past `outer` such walks.
    fn walk_inside_cube(&self, outer: usize) -> Space {
        // Where the planes or units span the launch, the levels inside the outermost cut a plane's
        // or a unit's box.
        let outermost = &self.partitioning.levels()[0];
        let mut enclosing = match (outermost.spans_launch(), outermost.coverage().scope()) {
            (true, Some(scope)) => self.partitioning.box_of(scope),
            _ => self.cube_box(),
        };
        let mut walks = self.partitioning.levels()[1..].iter().filter_map(|level| {
            let tile = level.child(&enclosing);
            enclosing = tile.clone();
            (level.coverage() == Coverage::Walk && level.tile(S).is_some()).then_some(tile)
        });
        walks
            .nth(outer)
            .expect("the partitioning walks no S inside its cubes")
    }

    /// The innermost box: what one instruction, or one unit's block, covers.
    pub fn leaf(&self) -> Space {
        self.partitioning.leaf()
    }

    /// Whether the tiles overhang `axis`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.partitioning.overhangs(axis)
    }

    /// Cubes one batch and key head of the problem spawns: one for every box of queries and
    /// heads, times the cubes sharing each.
    pub fn cubes(&self) -> usize {
        let (space, cube) = (self.partitioning.space(), self.cube_box());
        let boxes = |axis| space.extent(axis).div_ceil(cube.extent(axis));
        boxes(Q) * boxes(G) * self.cubes_along_keys.get()
    }

    /// The partitioning a kernel compiles against: every extent a launch varies — the batches, the
    /// key heads, the queries and the keys — left to the launch, and the group and the dims a row
    /// spans compiled in, which a row's registers and fragments are sized by.
    pub fn kernel(&self) -> Partitioning {
        Partitioning::new(
            self.partitioning
                .space()
                .clone()
                .with_dynamic(&[B, KV, Q, S]),
            self.partitioning.levels().to_vec(),
        )
    }

    /// cubek's partitioning, for what the tile engine reads of it: the launch and the binding.
    pub fn as_cubek(&self) -> &Partitioning {
        &self.partitioning
    }
}
