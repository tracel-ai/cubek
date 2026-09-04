//! What a kernel's levels become at launch: the [`Nest`] a kernel walks, the cube grid it
//! implies over the real extents, and the [`Launcher`] that binds it to a client for one kernel
//! launch. The launcher keeps the concrete (real-extent) nest alongside the kernel-form space, so
//! geometry and divisibility are always read off real extents and no call site can consume the
//! space too early.

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, CubeAxis, Geometry, Level, LevelCuts, Set, Space, StridedOperand,
    StridedTileSource, Unset,
};

/// A space and the levels a kernel walks it with, outermost first: what a launch sizes its grid
/// from. A blueprint lists its level methods into one; a kernel with no blueprint (a test, a
/// benchmark mapping) is handed its space and each of its levels from one, as comptime
/// arguments, so its loops state the levels the grid was read from.
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

    /// Add a level below the ones stated so far ([`Level::new`] over the space's axes).
    pub fn level(mut self, f: impl FnOnce(&mut LevelCuts)) -> Self {
        let axes: Vec<Axis> = self.space.axes().collect();
        self.levels.push(Level::new(&axes, f));
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

    /// Cube dimension `d` gets the instance count of whichever axis is `Spatial { Cube(d), .. }`,
    /// at any level, else 1.
    pub fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            self.instances(ComputeScope::Cube(CubeAxis::X)),
            self.instances(ComputeScope::Cube(CubeAxis::Y)),
            self.instances(ComputeScope::Cube(CubeAxis::Z)),
        )
    }

    /// `plane_size × plane_count`, plane length being the hardware's. `Unit` axes ride those
    /// lanes, so their instance product must be exactly `plane_size` or `1`; anything else idles
    /// or races lanes.
    pub fn cube_dim(&self, client: &Client) -> CubeDim {
        let plane_size = client.properties().hardware.plane_size_max;
        let lanes = self.instances(ComputeScope::Unit);
        assert!(
            lanes == 1 || lanes == plane_size,
            "Nest::cube_dim: Unit axes must partition exactly plane_size ({plane_size}) lanes, \
             got {lanes}"
        );
        CubeDim::new_2d(plane_size, self.instances(ComputeScope::Plane))
    }

    /// Product of instance counts over every axis riding `scope`, across every level, times the
    /// instance count of any work a level distributes as one on it ([`Work`](crate::Work)).
    fn instances(&self, scope: ComputeScope) -> u32 {
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

/// One launch's host-side bundle: the concrete nest (real extents, for geometry, overhang and
/// divisibility math) and the kernel-form space tile arguments project from.
pub struct Launcher<'c> {
    concrete: Nest,
    kernel: Space,
    client: &'c Client,
}

impl<'c> Launcher<'c> {
    /// The launcher of `nest`, whose extents are this launch's real ones, in kernel `form`.
    pub fn new(client: &'c Client, nest: &Nest, form: KernelForm<'_>) -> Self {
        let concrete = nest.clone();
        let kernel = match form {
            KernelForm::Dynamic => concrete.space.clone().all_dynamic(),
            KernelForm::Static => concrete.space.clone(),
            KernelForm::DynamicAlong(axes) => {
                // An axis the space does not have would be dropped by `with_dynamic`, leaving a
                // kernel specialized along the axis the caller meant to free.
                for &axis in axes {
                    assert!(
                        concrete.space.contains(axis),
                        "Launcher::new: {axis:?} is not an axis of this nest"
                    );
                }
                concrete.space.clone().with_dynamic(axes)
            }
        };
        Launcher {
            concrete,
            kernel,
            client,
        }
    }

    pub fn cube_count(&self) -> CubeCount {
        self.concrete.cube_count()
    }

    pub fn cube_dim(&self) -> CubeDim {
        self.concrete.cube_dim(self.client)
    }

    /// The kernel-form space tile arguments project from.
    pub fn space(&self) -> &Space {
        &self.kernel
    }

    /// The concrete nest, for overhang and divisibility decisions.
    pub fn concrete(&self) -> &Nest {
        &self.concrete
    }

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this launcher's
    /// kernel space, with automatic bounds checking derived from the concrete nest's overhang.
    pub fn arg(&self, binding: TensorBinding) -> StridedTileSource<'_, Set, Unset, Unset> {
        StridedOperand::source(binding)
            .space(&self.kernel)
            .concrete(&self.concrete)
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
            .concrete(&self.concrete)
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
        let Nest { space, levels } = &self.concrete;
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
