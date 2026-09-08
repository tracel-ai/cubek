//! One kernel launch: the [`Launcher`] binds a space, the grid the selector chose and the tiles
//! its operands are cut to to a client, and keeps the concrete (real-extent) space alongside the
//! kernel-form one, so geometry and divisibility are always read off real extents and no call
//! site can consume the space too early.

use cubecl::prelude::*;

use crate::{
    Axis, Geometry, Level, Partitioning, Set, Space, SpaceLaunch, StridedOperand,
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

/// One launch: a space, the grid the selector chose, the tiles the operands are cut to and the
/// axes that overhang, bound to a client. Every one of those is the blueprint's statement.
/// Geometry and divisibility are read off the concrete (real-extent) space, and tile arguments
/// project from the kernel-form one.
///
/// A launch that states its levels too ([`partitioned`](Launcher::partitioned)) can bind a
/// block-stored operand: its storage block has to be the tile of one of them
/// ([`Blocks`](crate::Blocks)). A kernel with no blueprint (a test, a benchmark mapping) is
/// [`implied`](Launcher::implied) by its levels instead, and keeps them to hand its loops one
/// each.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    concrete: Space,
    kernel: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    /// The tile every operand is cut to at the bottom, per axis; an axis not listed is whole.
    leaf: Vec<(Axis, usize)>,
    /// The axes some tile reaches past the end of, whose accesses are masked.
    overhangs: Vec<Axis>,
    /// The kernel's levels, outermost first, when the launch states them; what a block-stored
    /// operand's block is matched against. Empty for a launch that states only its leaf.
    levels: Vec<Level>,
}

impl Launcher {
    /// `space` on `grid`, its extents this launch's real ones, in kernel `form`. Refuses a cube
    /// the device cannot hold. State the leaf and the overhanging axes with
    /// [`leaf`](Launcher::leaf) and [`overhanging`](Launcher::overhanging).
    pub fn new(
        client: &Client,
        space: Space,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        let (cube_count, cube_dim) = grid;
        let max_units = client.properties().hardware.max_units_per_cube;
        assert!(
            cube_dim.num_elems() <= max_units,
            "Launcher::new: a cube of {} units, but the device holds at most {max_units}",
            cube_dim.num_elems()
        );
        let kernel = match form {
            KernelForm::Dynamic => space.clone().all_dynamic(),
            KernelForm::Static => space.clone(),
            KernelForm::DynamicAlong(axes) => {
                // An axis the space does not have would be dropped by `with_dynamic`, leaving a
                // kernel specialized along the axis the caller meant to free.
                for &axis in axes {
                    assert!(
                        space.contains(axis),
                        "Launcher::new: {axis:?} is not an axis of this space"
                    );
                }
                space.clone().with_dynamic(axes)
            }
        };
        Launcher {
            client: client.clone(),
            concrete: space,
            kernel,
            cube_count,
            cube_dim,
            leaf: Vec::new(),
            overhangs: Vec::new(),
            levels: Vec::new(),
        }
    }

    /// The tile every operand is cut to at the bottom: what a line width has to divide.
    pub fn leaf(mut self, leaf: &[(Axis, usize)]) -> Self {
        self.leaf = leaf.to_vec();
        self
    }

    /// The axes along which some tile reaches past the tensor, so every access is masked.
    pub fn overhanging(mut self, axes: &[Axis]) -> Self {
        self.overhangs = axes.to_vec();
        self
    }

    /// [`new`](Launcher::new) over the kernel's whole `partitioning`: the leaf and the overhangs
    /// read off its levels rather than stated beside them, and the levels kept, which is what
    /// lets a block-stored operand find the level its block is the tile of. The grid is still
    /// the blueprint's statement.
    pub fn partitioned(
        client: &Client,
        partitioning: Partitioning,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        let leaf = partitioning.leaf().extents();
        let overhangs = partitioning.overhanging();
        let (space, levels) = partitioning.into_parts();
        let mut launch = Launcher::new(client, space, grid, form)
            .leaf(&leaf)
            .overhanging(&overhangs);
        launch.levels = levels;
        launch
    }

    /// The launch `partitioning` implies, for a kernel with no blueprint to state one: as many
    /// cubes, planes and lanes as its levels deal to, the leaf they cut to, the axes they
    /// overhang. A third constructor, not `new`: a launch is stated, and this one reads off
    /// the levels what a blueprint would have stated, which only a test or a benchmark mapping
    /// wants.
    pub fn implied(client: &Client, partitioning: Partitioning, form: KernelForm<'_>) -> Self {
        let plane_size = client.properties().hardware.plane_size_max;
        let lanes = partitioning.lanes();
        assert!(
            lanes == 1 || lanes == plane_size,
            "Launcher::implied: Unit axes must partition exactly plane_size ({plane_size}) lanes, \
             got {lanes}"
        );
        let grid = (partitioning.cube_count(), partitioning.cube_dim(plane_size));
        Launcher::partitioned(client, partitioning, grid, form)
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

    /// The kernel's `space` argument: the kernel form, its dynamic extents sized by this launch.
    pub fn space_arg(&self) -> SpaceLaunch {
        self.kernel.launch_arg(&self.concrete)
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

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this launcher's
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
    pub fn vector_size(
        &self,
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
                "Launcher::vector_size: axis {axis:?} must label each operand's innermost dim"
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
        self.client
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
