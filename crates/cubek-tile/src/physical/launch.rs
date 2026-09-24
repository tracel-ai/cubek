//! One kernel launch, in two halves. A [`LaunchPlan`] is the geometry: the space, the grid the
//! selector chose and the tiles its operands are cut to, and every question that is arithmetic
//! over them. A [`Launcher`] is that plan bound to a client, which is where the device enters and
//! where a cube the device cannot hold is refused.
//!
//! The split is what lets a caller reason about a launch without launching it: a test, a selector
//! weighing two grids, a benchmark mapping. None of them has a kernel to dispatch, and none of
//! them should have to satisfy a device to ask what a launch's geometry would be.

use cubecl::ir::OpaqueType;
use cubecl::prelude::*;

use crate::{
    Axis, Geometry, Level, Partitioning, PartitioningLaunch, Set, Space, StridedOperand,
    StridedTileSource, Unset,
};

/// Which extents the compiled kernel reads at runtime. Every one (`Dynamic`) makes one compiled
/// kernel serve every shape; none (`Static`) specializes the kernel to this launch's extents;
/// `DynamicAlong` frees only the listed axes, which specializes the loops along the others and
/// serves an axis no operand can state the size of ([`Tile::witnesses`](crate::Tile::witnesses)).
#[derive(Clone, Copy, Debug)]
pub enum KernelForm<'a> {
    Dynamic,
    Static,
    DynamicAlong(&'a [Axis]),
}

/// A launch's geometry, with no device attached: a space, the grid the selector chose, the tiles
/// the operands are cut to and the axes that overhang. Every one of those is the blueprint's
/// statement, and every question here is arithmetic over them.
///
/// Geometry and divisibility are read off the concrete (real-extent) space, and tile arguments
/// project from the kernel-form one.
///
/// Nothing here reads a device. Where a plan needs a device fact it is handed one — a plane size
/// to [`implied`](LaunchPlan::implied), a client to [`vector_size`](LaunchPlan::vector_size) —
/// rather than carrying a client of its own. Bind it with [`Launcher::bind`] to launch it.
#[derive(Clone)]
pub struct LaunchPlan {
    concrete: Space,
    kernel: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    /// The tile every operand is cut to at the bottom, per axis; an axis not listed is whole.
    leaf: Vec<(Axis, usize)>,
    /// The axes some tile reaches past the end of, whose accesses are masked.
    overhangs: Vec<Axis>,
    /// The kernel's levels, outermost first, when the launch states them; what a storage-tiled
    /// operand's storage tile is matched against. Empty for a launch that states only its leaf.
    levels: Vec<Level>,
}

impl LaunchPlan {
    /// `space` on `grid`, its extents this launch's real ones, in kernel `form`, cut by no level
    /// and overhanging nowhere. [`partitioned`](LaunchPlan::partitioned) states both off a
    /// partitioning's levels.
    pub fn new(space: Space, grid: (CubeCount, CubeDim), form: KernelForm<'_>) -> Self {
        let (cube_count, cube_dim) = grid;
        let kernel = match form {
            KernelForm::Dynamic => space.clone().all_dynamic(),
            KernelForm::Static => space.clone(),
            KernelForm::DynamicAlong(axes) => {
                // An axis the space does not have would be dropped by `with_dynamic`, leaving a
                // kernel specialized along the axis the caller meant to free.
                for &axis in axes {
                    assert!(
                        space.contains(axis),
                        "LaunchPlan::new: {axis:?} is not an axis of this space"
                    );
                }
                space.clone().with_dynamic(axes)
            }
        };
        LaunchPlan {
            concrete: space,
            kernel,
            cube_count,
            cube_dim,
            leaf: Vec::new(),
            overhangs: Vec::new(),
            levels: Vec::new(),
        }
    }

    /// [`new`](LaunchPlan::new) over the kernel's whole `partitioning`: the leaf and the overhangs
    /// read off its levels rather than stated beside them, and the levels kept, which is what lets
    /// a storage-tiled operand find the level its storage tile is the tile of. The grid is still
    /// the blueprint's statement.
    pub fn partitioned(
        partitioning: Partitioning,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        let leaf = partitioning.leaf().extents();
        let overhangs = partitioning.overhanging();
        let Partitioning { space, levels } = partitioning;
        LaunchPlan {
            leaf,
            overhangs,
            levels,
            ..LaunchPlan::new(space, grid, form)
        }
    }

    /// The plan `partitioning` implies, for a kernel with no blueprint to state one: as many
    /// cubes, planes and lanes as its levels deal to, the leaf they cut to, the axes they
    /// overhang. A third constructor, not `new`: a launch is stated, and this one reads off the
    /// levels what a blueprint would have stated, which only a test or a benchmark mapping wants.
    ///
    /// `plane_size` is the device's, and is stated rather than read so a caller can ask what this
    /// partitioning implies on a plane it does not have in front of it.
    pub fn implied(partitioning: Partitioning, plane_size: u32, form: KernelForm<'_>) -> Self {
        let lanes = partitioning.lanes();
        assert!(
            lanes == 1 || lanes == plane_size,
            "LaunchPlan::implied: Unit axes must partition exactly plane_size ({plane_size}) \
             lanes, got {lanes}"
        );
        let grid = (partitioning.cube_count(), partitioning.cube_dim(plane_size));
        LaunchPlan::partitioned(partitioning, grid, form)
    }

    /// How many planes this plan sets aside to fill a walk's stages rather than to compute.
    /// The two roles meet on a barrier, which is a device fact, so it is [`Launcher::bind`] that
    /// refuses a device carrying no barrier type.
    fn fillers(&self) -> u32 {
        self.levels.iter().map(|level| level.fillers() as u32).sum()
    }

    pub fn cube_count(&self) -> CubeCount {
        self.cube_count.clone()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.cube_dim
    }

    /// The leaf edge along `axis`: the stated tile, or the whole extent where none was.
    fn leaf_edge(&self, axis: Axis) -> usize {
        self.leaf
            .iter()
            .find(|&&(a, _)| a == axis)
            .map(|&(_, edge)| edge)
            .unwrap_or_else(|| self.concrete.extent(axis))
    }

    /// The concrete space: this launch's real extents.
    pub fn space(&self) -> &Space {
        &self.concrete
    }

    /// The kernel-form space tile arguments project from.
    pub fn kernel_space(&self) -> &Space {
        &self.kernel
    }

    /// The kernel's `space` argument: the kernel-form space, its dynamic extents sized by this
    /// launch, under the levels the launch states. What `for cube in space` iterates.
    pub fn partitioning_arg(&self) -> PartitioningLaunch {
        PartitioningLaunch::new(
            self.kernel.space_launch(&self.concrete),
            self.levels.clone(),
        )
    }

    /// The levels a partitioned or implied launch states, outermost first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// The partitioning this launch states: its concrete space with those levels. A launch that
    /// stated only its leaf has no levels, so this is its space uncut.
    pub fn partitioning(&self) -> Partitioning {
        Partitioning::new(self.concrete.clone(), self.levels.clone())
    }

    /// Level `i` of a partitioned or implied launch, outermost first: what a kernel states its
    /// `i`-th loop with.
    pub fn level(&self, i: usize) -> Level {
        self.levels[i].clone()
    }

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this plan's
    /// kernel space, with automatic bounds checking derived from the concrete nest's overhang.
    pub fn arg(&self, binding: TensorBinding) -> StridedTileSource<'_, Set, Unset, Unset> {
        StridedOperand::source(binding)
            .space(&self.kernel)
            .concrete(&self.concrete, &self.overhangs)
            .cube_units(self.cube_dim().num_elems() as usize)
            .levels(&self.levels)
    }

    /// [`arg`](Self::arg) over a stated geometry rather than a binding, for an operand with no
    /// tensor: the destination a fused store writes through
    /// ([`Tile::of_sink`](crate::Tile::of_sink)) or the producer a fused read comes from
    /// ([`Tile::of_source`](crate::Tile::of_source)). `geometry` is the physical extents and
    /// strides the operand *would* have had; everything else is settled exactly as for a bound
    /// operand, since this is the same builder.
    ///
    /// End it with [`build_spec`](StridedTileSource::build_spec), not
    /// [`build`](StridedTileSource::build): there is no tensor to ship, and the *settled* geometry
    /// comes back beside the spec. The two part company where a broadcast batch dim is dropped,
    /// which is why the settled one travels rather than the call site reproducing the drop.
    pub fn geometry(&self, geometry: &Geometry) -> StridedTileSource<'_, Set, Unset, Unset> {
        StridedTileSource::<Unset, Unset, Unset>::of_geometry(geometry)
            .space(&self.kernel)
            .concrete(&self.concrete, &self.overhangs)
            .cube_units(self.cube_dim().num_elems() as usize)
            .levels(&self.levels)
    }

    /// The widest `Vector<E, v>` line every operand can be served in along `axis`: one width for
    /// all of them, since a kernel reading one operand's lines writes the other's. Each
    /// `(geometry, subspace)` must be unchecked and innermost-contiguous, and the width must
    /// divide each inner extent, every coarser stride and the axis's leaf tile edge; `1`
    /// otherwise. Takes a [`Geometry`] rather than a binding so an operand with no tensor
    /// constrains the shared width like any other.
    ///
    /// `client` is the only device fact this asks for: which line widths its I/O is optimized
    /// for. Everything else is read off the plan.
    pub fn vector_size(
        &self,
        client: &Client,
        axis: Axis,
        operands: &[(&Geometry, &[Axis])],
        type_size: usize,
    ) -> usize {
        // The width gates below test the physical innermost dim, so `axis` must be the label
        // of every operand's innermost buffer dim (`subspace` labels repeat level-major).
        for (_, subspace) in operands {
            assert_eq!(
                subspace.last(),
                Some(&axis),
                "LaunchPlan::vector_size: axis {axis:?} must label each operand's innermost dim"
            );
        }
        // The one gate that is about the tiles rather than the geometry: a masked access reports
        // its length in lines and would wrongly clip, so an overhanging subspace is served scalar
        // whatever its extents and strides would allow. `serves_lines` below answers the rest.
        let masked = operands
            .iter()
            .any(|(_, subspace)| subspace.iter().any(|a| self.overhangs.contains(a)));
        if masked {
            return 1;
        }
        let leaf = self.leaf_edge(axis);
        client
            .io_optimized_vector_sizes(type_size)
            .filter(|&v| {
                leaf.is_multiple_of(v)
                    // The same gates `Geometry::serves_lines` refuses a stated width on: the
                    // innermost extent counts in lines and every coarser stride re-expresses
                    // as `stride / v`, which truncates when `v` does not divide it.
                    && operands.iter().all(|(g, _)| g.serves_lines(v).is_ok())
            })
            .max()
            .unwrap_or(1)
    }
}

/// One launch: a [`LaunchPlan`] bound to the client it will run on. Binding is where the device
/// is first known, so it is where a cube the device cannot hold, and a walk whose two roles have
/// no barrier to meet on, are refused — on the host, before anything is dispatched.
///
/// Every geometry question is the plan's, and delegated to it. A caller that only wants to ask
/// those questions wants a [`LaunchPlan`] and no client.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    plan: LaunchPlan,
}

impl Launcher {
    /// `plan` on `client`. Refuses a cube the device cannot hold, and a plan that sets planes
    /// aside to fill a walk's stages on a device carrying no barrier type.
    pub fn bind(client: &Client, plan: LaunchPlan) -> Self {
        let max_units = client.properties().hardware.max_units_per_cube;
        assert!(
            plan.cube_dim().num_elems() <= max_units,
            "Launcher::bind: a cube of {} units, but the device holds at most {max_units}",
            plan.cube_dim().num_elems()
        );
        // The two roles meet on a barrier and nowhere else, so a device that carries no barrier
        // type would run two loops with no rendezvous between them. Refused here, on the host,
        // and not where the slot is allocated: a refusal at expansion fires on a worker thread,
        // where nothing sees it and the launch returns zeros.
        let fillers = plan.fillers();
        assert!(
            fillers == 0
                || client
                    .properties()
                    .features
                    .types
                    .opaque
                    .contains(&OpaqueType::Barrier),
            "Launcher: {fillers} plane(s) are set aside to fill a walk's stages, and this device \
             carries no barrier type for the two roles to meet on"
        );
        Launcher {
            client: client.clone(),
            plan,
        }
    }

    /// [`LaunchPlan::new`] bound to `client`.
    pub fn new(
        client: &Client,
        space: Space,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        Launcher::bind(client, LaunchPlan::new(space, grid, form))
    }

    /// [`LaunchPlan::partitioned`] bound to `client`.
    pub fn partitioned(
        client: &Client,
        partitioning: Partitioning,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        Launcher::bind(client, LaunchPlan::partitioned(partitioning, grid, form))
    }

    /// [`LaunchPlan::implied`] on `client`'s plane size, bound to it.
    pub fn implied(client: &Client, partitioning: Partitioning, form: KernelForm<'_>) -> Self {
        let plane_size = client.properties().hardware.plane_size_max;
        Launcher::bind(client, LaunchPlan::implied(partitioning, plane_size, form))
    }

    /// The geometry this launch runs, free of the client it is bound to.
    pub fn plan(&self) -> &LaunchPlan {
        &self.plan
    }

    /// The client this launch runs on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    // ---- the plan's questions, asked through the launch it is bound into ----

    pub fn cube_count(&self) -> CubeCount {
        self.plan.cube_count()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.plan.cube_dim()
    }

    /// See [`LaunchPlan::space`].
    pub fn space(&self) -> &Space {
        self.plan.space()
    }

    /// See [`LaunchPlan::kernel_space`].
    pub fn kernel_space(&self) -> &Space {
        self.plan.kernel_space()
    }

    /// See [`LaunchPlan::partitioning_arg`].
    pub fn partitioning_arg(&self) -> PartitioningLaunch {
        self.plan.partitioning_arg()
    }

    /// See [`LaunchPlan::levels`].
    pub fn levels(&self) -> &[Level] {
        self.plan.levels()
    }

    /// See [`LaunchPlan::partitioning`].
    pub fn partitioning(&self) -> Partitioning {
        self.plan.partitioning()
    }

    /// See [`LaunchPlan::level`].
    pub fn level(&self, i: usize) -> Level {
        self.plan.level(i)
    }

    /// See [`LaunchPlan::arg`].
    pub fn arg(&self, binding: TensorBinding) -> StridedTileSource<'_, Set, Unset, Unset> {
        self.plan.arg(binding)
    }

    /// See [`LaunchPlan::geometry`].
    pub fn geometry(&self, geometry: &Geometry) -> StridedTileSource<'_, Set, Unset, Unset> {
        self.plan.geometry(geometry)
    }

    /// See [`LaunchPlan::vector_size`], asked of the client this launch is bound to.
    pub fn vector_size(
        &self,
        axis: Axis,
        operands: &[(&Geometry, &[Axis])],
        type_size: usize,
    ) -> usize {
        self.plan
            .vector_size(&self.client, axis, operands, type_size)
    }
}
