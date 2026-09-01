//! What a [`Space`] becomes at launch: the cube grid its partitioner tree implies, and the
//! [`Launcher`] that binds it to a client for one kernel launch. The launcher keeps the
//! concrete (real-extent) space alongside the derived kernel-form (dynamic) one, so geometry
//! and divisibility are always read off real extents and no call site can consume the space
//! too early.

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, CubeAxis, Geometry, Operand, Set, Space, StridedOperand, StridedTileSource,
    Unset,
};

impl Space {
    /// Cube dimension `d` gets the instance count of whichever axis is
    /// `Spatial { Cube(d), .. }`, at any level of the tree, else 1.
    pub fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            instances_count(self, ComputeScope::Cube(CubeAxis::X)),
            instances_count(self, ComputeScope::Cube(CubeAxis::Y)),
            instances_count(self, ComputeScope::Cube(CubeAxis::Z)),
        )
    }

    /// `plane_size × plane_count`, plane length being the hardware's. `Unit` axes ride those
    /// lanes, so their instance product must be exactly `plane_size` or `1`; anything else idles
    /// or races lanes. A deferred `PlaneLanes` count panics here: launch through
    /// [`launcher`](Space::launcher), which stamps it.
    pub fn cube_dim<R: Runtime>(&self, client: &ComputeClient<R>) -> CubeDim {
        let plane_size = client.properties().hardware.plane_size_max;
        let lanes = instances_count(self, ComputeScope::Unit);
        assert!(
            lanes == 1 || lanes == plane_size,
            "cube_dim: Unit axes must partition exactly plane_size ({plane_size}) lanes, got {lanes}"
        );
        CubeDim::new_2d(plane_size, instances_count(self, ComputeScope::Plane))
    }
}

/// Product of instance counts over every axis riding `scope`, across the whole partitioner tree,
/// times the instance count of any work a level distributes as one on it ([`Work`]).
fn instances_count(space: &Space, scope: ComputeScope) -> u32 {
    let mut total = 1u32;
    let mut level = space.clone();
    while !level.is_final() {
        // Work distributed as one rides its scope whole rather than through any one of its
        // axes, so its instance count is the dim's and no axis of it contributes.
        if let Some(work) = level.partitioner().work()
            && work.scope() == scope
        {
            total *= work.instances() as u32;
        }
        for axis in level.axes() {
            let dist = level.partitioner().distribution(axis);
            if dist.scope() == Some(scope) {
                // `count` is `ceil`, so an indivisible axis adds the instance for its
                // partial tile.
                total *= dist.coverage().instances(level.count(axis)) as u32;
            }
        }
        level = level.divide();
    }
    total
}

/// One launch's host-side bundle: the concrete space (real extents, for geometry, overhang and
/// divisibility math) and the kernel-form space tile arguments project from.
pub struct Launcher<'c, R: Runtime> {
    concrete: Space,
    kernel: Space,
    client: &'c ComputeClient<R>,
}

impl Space {
    /// Creates a [`Launcher`] with all kernel space axes marked dynamic, so one compiled kernel
    /// serves arbitrary shapes, resolving `Unit` lane counts from the device `plane_size`. Use
    /// [`launcher_over`](Self::launcher_over) to keep specific axes static.
    pub fn launcher<R: Runtime>(self, client: &ComputeClient<R>) -> Launcher<'_, R> {
        let plane_size = client.properties().hardware.plane_size_max as usize;
        let concrete = self.resolve_lanes(plane_size);
        let kernel = concrete.clone().all_dynamic();
        Launcher::new(concrete, kernel, client)
    }

    /// Creates a [`Launcher`] where only the `dynamic` axes have dynamic extents, every other
    /// axis staying comptime. Specializes kernel loops along an axis, and serves one no operand
    /// can state the size of ([`Tile::witnesses`](crate::Tile::witnesses)); `&[]` is fully static.
    pub fn launcher_over<'c, R: Runtime>(
        self,
        client: &'c ComputeClient<R>,
        dynamic: &[Axis],
    ) -> Launcher<'c, R> {
        // An axis the space does not have would be dropped by `with_dynamic`, leaving a kernel
        // specialized along the axis the caller meant to free.
        for &axis in dynamic {
            assert!(
                self.contains(axis),
                "Space::launcher_over: {axis:?} is not an axis of this space"
            );
        }
        let plane_size = client.properties().hardware.plane_size_max as usize;
        let concrete = self.resolve_lanes(plane_size);
        let kernel = concrete.clone().with_dynamic(dynamic);
        Launcher::new(concrete, kernel, client)
    }
}

impl<'c, R: Runtime> Launcher<'c, R> {
    fn new(concrete: Space, kernel: Space, client: &'c ComputeClient<R>) -> Self {
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

    /// The kernel-form (fully dynamic) space tile arguments project from.
    pub fn space(&self) -> &Space {
        &self.kernel
    }

    /// The concrete space, for overhang and divisibility decisions.
    pub fn concrete(&self) -> &Space {
        &self.concrete
    }

    /// Starts configuring a tile operand builder ([`StridedTileSource`]) bound to this launcher's
    /// kernel space, with automatic bounds checking derived from the concrete space overhang.
    pub fn arg(&self, binding: TensorBinding<R>) -> StridedTileSource<'_, Set, Unset, Unset, R> {
        StridedOperand::source(binding)
            .space(&self.kernel)
            .concrete(&self.concrete)
            .cube_units(self.cube_dim().num_elems() as usize)
    }

    /// [`arg`](Self::arg) driven by a sealed [`Operand`]: the subspace is the operand's axes
    /// and the per-level residences its stages, stated once where the levels were declared, so
    /// neither can drift from the space the way a hand-passed array can.
    pub fn bind<'a>(
        &'a self,
        operand: &'a Operand,
        binding: TensorBinding<R>,
    ) -> StridedTileSource<'a, Set, Set, Unset, R> {
        self.arg(binding).subspace(operand.axes()).operand(operand)
    }

    /// [`bind`](Self::bind) over a stated geometry rather than a binding, for an operand with no
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
    pub fn bind_geometry<'a>(
        &'a self,
        operand: &'a Operand,
        geometry: &Geometry,
    ) -> StridedTileSource<'a, Set, Set, Unset, R> {
        StridedTileSource::<Unset, Unset, Unset, R>::of_geometry(geometry)
            .space(&self.kernel)
            .concrete(&self.concrete)
            .cube_units(self.cube_dim().num_elems() as usize)
            .subspace(operand.axes())
            .operand(operand)
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
        // The one gate that is about the space rather than the geometry: a masked access reports
        // its length in lines and would wrongly clip, so an overhanging subspace is served scalar
        // whatever its extents and strides would allow. `serves_lines` below answers the rest.
        let masked = operands
            .iter()
            .any(|(_, subspace)| subspace.iter().any(|&a| self.concrete.overhangs(a)));
        if masked {
            return 1;
        }
        let leaf = self.concrete.final_space().extent(axis);
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
