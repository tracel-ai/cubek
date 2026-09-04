//! What a kernel's levels become at launch: the cube grid they imply over the real extents, and
//! the [`Launcher`] that binds them to a client for one kernel launch. The launcher keeps the
//! concrete (real-extent) space alongside the kernel-form (dynamic) one, so geometry and
//! divisibility are always read off real extents and no call site can consume the space too
//! early.

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, CubeAxis, Geometry, Level, Set, Space, StridedOperand, StridedTileSource,
    Unset, leaf,
};

/// Cube dimension `d` gets the instance count of whichever axis is `Spatial { Cube(d), .. }`, at
/// any level, else 1.
pub fn cube_count(space: &Space, levels: &[Level]) -> CubeCount {
    CubeCount::Static(
        instances_count(space, levels, ComputeScope::Cube(CubeAxis::X)),
        instances_count(space, levels, ComputeScope::Cube(CubeAxis::Y)),
        instances_count(space, levels, ComputeScope::Cube(CubeAxis::Z)),
    )
}

/// `plane_size × plane_count`, plane length being the hardware's. `Unit` axes ride those lanes,
/// so their instance product must be exactly `plane_size` or `1`; anything else idles or races
/// lanes.
pub fn cube_dim(space: &Space, levels: &[Level], client: &Client) -> CubeDim {
    let plane_size = client.properties().hardware.plane_size_max;
    let lanes = instances_count(space, levels, ComputeScope::Unit);
    assert!(
        lanes == 1 || lanes == plane_size,
        "cube_dim: Unit axes must partition exactly plane_size ({plane_size}) lanes, got {lanes}"
    );
    CubeDim::new_2d(
        plane_size,
        instances_count(space, levels, ComputeScope::Plane),
    )
}

/// Product of instance counts over every axis riding `scope`, across every level, times the
/// instance count of any work a level distributes as one on it ([`Work`](crate::Work)).
fn instances_count(space: &Space, levels: &[Level], scope: ComputeScope) -> u32 {
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

/// Whether `axis` overhangs its tiling: some level's edge fails to divide the extent handed to
/// it (the top extent at the first level, the parent edge below), leaving a partial tile that
/// needs masking. Host-side, on the concrete (real-extent) space; a
/// [`Dynamic`](crate::Extent) axis panics.
pub fn overhangs(space: &Space, levels: &[Level], axis: Axis) -> bool {
    assert!(
        !space.is_dynamic(axis),
        "overhangs: axis {axis:?} is Dynamic; ask the concrete space, not the kernel-form one"
    );
    let mut space = space.clone();
    for level in levels {
        if level.overhangs(&space, axis) {
            return true;
        }
        space = level.child(&space);
    }
    false
}

/// The concrete (real-extent) space of a launch and the levels a kernel walks it with: what a
/// launch-side derivation (overhang, the leaf edge, the block a scheme is checked against) reads.
#[derive(Clone, Copy)]
pub struct Concrete<'a> {
    pub space: &'a Space,
    pub levels: &'a [Level],
}

impl Concrete<'_> {
    pub fn overhangs(&self, axis: Axis) -> bool {
        overhangs(self.space, self.levels, axis)
    }

    pub fn contains(&self, axis: Axis) -> bool {
        self.space.contains(axis)
    }
}

/// One launch's host-side bundle: the concrete space (real extents, for geometry, overhang and
/// divisibility math), the levels, and the kernel-form space tile arguments project from.
pub struct Launcher<'c> {
    concrete: Space,
    kernel: Space,
    levels: Vec<Level>,
    client: &'c Client,
}

impl<'c> Launcher<'c> {
    /// The launcher for the `levels` a kernel states, outermost first, over the real `extents`
    /// of this launch (whose order is the axes' canonical order). The concrete twin takes the
    /// extents for geometry, overhang and the grid; the kernel-form space keeps every extent
    /// [`Dynamic`](crate::Extent), so one compiled kernel serves every shape.
    pub fn new(client: &'c Client, extents: &[(Axis, usize)], levels: &[Level]) -> Self {
        let axes: Vec<Axis> = extents.iter().map(|&(a, _)| a).collect();
        Launcher {
            concrete: Space::new(extents),
            kernel: Space::dynamic(&axes),
            levels: levels.to_vec(),
            client,
        }
    }

    /// [`new`](Launcher::new) for a kernel whose extents are all static: the kernel form is the
    /// concrete space itself.
    pub fn over_static(client: &'c Client, extents: &[(Axis, usize)], levels: &[Level]) -> Self {
        Launcher::over(client, extents, &[], levels)
    }

    /// The launcher whose kernel form frees only the `dynamic` axes, every other extent staying
    /// comptime: specializes kernel loops along an axis, and serves one no operand can state the
    /// size of ([`Tile::witnesses`](crate::Tile::witnesses)); `&[]` is fully static.
    pub fn over(
        client: &'c Client,
        extents: &[(Axis, usize)],
        dynamic: &[Axis],
        levels: &[Level],
    ) -> Self {
        let concrete = Space::new(extents);
        // An axis the space does not have would be dropped by `with_dynamic`, leaving a kernel
        // specialized along the axis the caller meant to free.
        for &axis in dynamic {
            assert!(
                concrete.contains(axis),
                "Launcher::over: {axis:?} is not an axis of this space"
            );
        }
        Launcher {
            kernel: concrete.clone().with_dynamic(dynamic),
            concrete,
            levels: levels.to_vec(),
            client,
        }
    }

    pub fn cube_count(&self) -> CubeCount {
        cube_count(&self.concrete, &self.levels)
    }

    pub fn cube_dim(&self) -> CubeDim {
        cube_dim(&self.concrete, &self.levels, self.client)
    }

    /// The kernel-form space tile arguments project from.
    pub fn space(&self) -> &Space {
        &self.kernel
    }

    /// The kernel-form space with the levels: what a kernel taking its nest as a comptime
    /// argument is handed.
    pub fn nest(&self) -> Nest {
        Nest::new(self.kernel.clone(), self.levels.clone())
    }

    /// The concrete space and the levels, for overhang and divisibility decisions.
    pub fn concrete(&self) -> Concrete<'_> {
        Concrete {
            space: &self.concrete,
            levels: &self.levels,
        }
    }

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this launcher's
    /// kernel space, with automatic bounds checking derived from the concrete space overhang.
    pub fn arg(&self, binding: TensorBinding) -> StridedTileSource<'_, Set, Unset, Unset> {
        StridedOperand::source(binding)
            .space(&self.kernel)
            .concrete(self.concrete())
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
            .concrete(self.concrete())
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
        let masked = operands.iter().any(|(_, subspace)| {
            subspace
                .iter()
                .any(|&a| overhangs(&self.concrete, &self.levels, a))
        });
        if masked {
            return 1;
        }
        let leaf = leaf(&self.concrete, &self.levels).extent(axis);
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

/// A space and the levels a kernel walks it with, stated together on the host: what a launch
/// with no blueprint of its own lists (a test, a benchmark mapping). The kernel takes the list
/// as a comptime argument and states each loop with one of its levels; the launch reads the
/// grid off the same list.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Nest {
    pub space: Space,
    pub levels: Vec<Level>,
}

impl Nest {
    /// `space` walked with `levels`, outermost first.
    pub fn new(space: Space, levels: Vec<Level>) -> Self {
        Nest { space, levels }
    }

    /// Static extents, no level yet.
    pub fn over(extents: &[(Axis, usize)]) -> Self {
        Nest {
            space: Space::new(extents),
            levels: Vec::new(),
        }
    }

    /// Every extent [`Dynamic`](crate::Extent), no level yet.
    pub fn dynamic(axes: &[Axis]) -> Self {
        Nest {
            space: Space::dynamic(axes),
            levels: Vec::new(),
        }
    }

    /// Add a level below the ones stated so far ([`Level::cuts`] over the space's axes).
    pub fn level(mut self, f: impl FnOnce(&mut crate::LevelCuts)) -> Self {
        let axes: Vec<Axis> = self.space.axes().collect();
        self.levels.push(Level::cuts(&axes, f));
        self
    }

    /// Level `i`, outermost first: what a kernel states its `i`-th loop with.
    pub fn at(&self, i: usize) -> Level {
        self.levels[i].clone()
    }

    /// The levels below the `i`-th: what an accumulator opened `i` levels down is shaped by.
    pub fn below(&self, i: usize) -> &[Level] {
        &self.levels[i..]
    }

    pub fn cube_count(&self) -> CubeCount {
        cube_count(&self.space, &self.levels)
    }

    pub fn cube_dim(&self, client: &Client) -> CubeDim {
        cube_dim(&self.space, &self.levels, client)
    }

    /// The launcher over this nest: the kernel form frees every axis ([`Launcher::new`]), the
    /// concrete space keeps the nest's extents.
    pub fn launcher<'c>(&self, client: &'c Client) -> Launcher<'c> {
        let extents: Vec<(Axis, usize)> = self
            .space
            .axes()
            .map(|a| (a, self.space.extent(a)))
            .collect();
        Launcher::new(client, &extents, &self.levels)
    }

    /// The launcher whose kernel form frees only the `dynamic` axes ([`Launcher::over`]).
    pub fn launcher_over<'c>(&self, client: &'c Client, dynamic: &[Axis]) -> Launcher<'c> {
        let extents: Vec<(Axis, usize)> = self
            .space
            .axes()
            .map(|a| (a, self.space.extent(a)))
            .collect();
        Launcher::over(client, &extents, dynamic, &self.levels)
    }
}
