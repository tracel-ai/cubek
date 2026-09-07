//! One kernel launch: the [`Launcher`] binds a space, the levels a kernel walks it with and the
//! grid the selector chose to a client, and keeps the concrete (real-extent) space alongside the
//! kernel-form one, so geometry and divisibility are always read off real extents and no call
//! site can consume the space too early.

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, CubeAxis, Geometry, Level, Set, Space, SpaceLaunch, StridedOperand,
    StridedTileSource, Unset,
};

/// The grid `levels` over `space` deal to: cube dimension `d` gets the instance count of
/// whichever axis rides `Cube(d)`, at any level; the cube is `plane_size × plane_count`. `Unit`
/// axes ride the plane's lanes, so their instance product must be exactly `plane_size` or `1`;
/// anything else idles or races lanes.
pub(crate) fn implied_grid(
    client: &Client,
    space: &Space,
    levels: &[Level],
) -> (CubeCount, CubeDim) {
    let plane_size = client.properties().hardware.plane_size_max;
    let lanes = instances(space, levels, ComputeScope::Unit);
    assert!(
        lanes == 1 || lanes == plane_size,
        "Launcher: Unit axes must partition exactly plane_size ({plane_size}) lanes, got {lanes}"
    );
    (
        cube_count(space, levels),
        CubeDim::new_2d(plane_size, instances(space, levels, ComputeScope::Plane)),
    )
}

/// Cube dimension `d` gets the instance count of whichever axis is `Spatial { Cube(d), .. }`,
/// at any level of `levels` over `space`, else 1.
pub(crate) fn cube_count(space: &Space, levels: &[Level]) -> CubeCount {
    CubeCount::Static(
        instances(space, levels, ComputeScope::Cube(CubeAxis::X)),
        instances(space, levels, ComputeScope::Cube(CubeAxis::Y)),
        instances(space, levels, ComputeScope::Cube(CubeAxis::Z)),
    )
}

/// Product of instance counts over every axis riding `scope`, across every level, times the
/// instance count of any work a level distributes as one on it ([`Work`](crate::Work)).
fn instances(space: &Space, levels: &[Level], scope: ComputeScope) -> u32 {
    let mut total = 1u32;
    let mut space = space.clone();
    for level in levels {
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

/// One launch: a space, the levels a kernel walks it with (outermost first), the grid it runs
/// on, and the client. The grid is the selector's decision, handed in by the blueprint; the
/// levels are checked against it and read for the operand gates (overhang, line width, scale
/// blocks), never for the grid. Geometry and divisibility are read off the concrete
/// (real-extent) space, and tile arguments project from the kernel-form one.
#[derive(Clone)]
pub struct Launcher {
    client: Client,
    concrete: Space,
    levels: Vec<Level>,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    kernel: Space,
}

impl Launcher {
    /// `space` walked with `levels` on `grid`, its extents this launch's real ones, in kernel
    /// `form`. Refuses a grid the levels do not deal to, and a cube the device cannot hold.
    pub fn new(
        client: &Client,
        space: Space,
        levels: Vec<Level>,
        grid: (CubeCount, CubeDim),
        form: KernelForm<'_>,
    ) -> Self {
        let (cube_count, cube_dim) = grid;
        let (implied_count, implied_dim) = implied_grid(client, &space, &levels);
        let counts_agree = match (&cube_count, &implied_count) {
            (CubeCount::Static(x, y, z), CubeCount::Static(ix, iy, iz)) => {
                (x, y, z) == (ix, iy, iz)
            }
            _ => false,
        };
        assert!(
            counts_agree && cube_dim == implied_dim,
            "Launcher::new: the levels deal to {implied_count:?} cubes of {implied_dim:?}, but \
             the launch is {cube_count:?} cubes of {cube_dim:?}"
        );
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
            levels,
            cube_count,
            cube_dim,
            kernel,
        }
    }

    /// The launch `levels` imply, for a kernel with no blueprint to decide its grid (a test, a
    /// benchmark mapping): as many cubes, planes and lanes as the levels deal to.
    pub fn implied(
        client: &Client,
        space: Space,
        levels: Vec<Level>,
        form: KernelForm<'_>,
    ) -> Self {
        let grid = implied_grid(client, &space, &levels);
        Launcher::new(client, space, levels, grid, form)
    }

    pub fn cube_count(&self) -> CubeCount {
        self.cube_count.clone()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.cube_dim
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

    /// The levels, outermost first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Level `i`, outermost first: what a kernel states its `i`-th loop with.
    pub fn level(&self, i: usize) -> Level {
        self.levels[i].clone()
    }

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this launcher's
    /// kernel space, with automatic bounds checking derived from the concrete nest's overhang.
    pub fn arg(&self, binding: TensorBinding) -> StridedTileSource<'_, Set, Unset, Unset> {
        StridedOperand::source(binding)
            .space(&self.kernel)
            .concrete(&self.concrete, &self.levels)
            .cube_units(self.cube_dim().num_elems() as usize)
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
            .concrete(&self.concrete, &self.levels)
            .cube_units(self.cube_dim().num_elems() as usize)
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
        // The one gate that is about the levels rather than the geometry: a masked access reports
        // its length in lines and would wrongly clip, so an overhanging subspace is served scalar
        // whatever its extents and strides would allow. `serves_lines` below answers the rest.
        let (space, levels) = (&self.concrete, &self.levels);
        let masked = operands
            .iter()
            .any(|(_, subspace)| subspace.iter().any(|&a| space.overhangs(levels, a)));
        if masked {
            return 1;
        }
        let leaf = space.leaf(levels).extent(axis);
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
