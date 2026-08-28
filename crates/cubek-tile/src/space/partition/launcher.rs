//! The [`Launcher`]: a concrete [`Space`] bound to a client for one kernel launch. It keeps
//! the concrete (real-extent) space alongside the derived kernel-form (dynamic) one, so
//! geometry and divisibility are always read off real extents and no call site can consume
//! the space too early.

use cubecl::prelude::*;

use crate::{Axis, Geometry, Operand, Set, Space, StridedOperand, StridedTileSource, Unset};

/// One launch's host-side bundle: the concrete space (real extents, for geometry, overhang and
/// divisibility math) and the kernel-form space tile arguments project from.
pub struct Launcher<'c, R: Runtime> {
    concrete: Space,
    kernel: Space,
    client: &'c ComputeClient<R>,
}

impl Space {
    /// Creates a [`Launcher`] with all kernel space axes marked dynamic, allowing one compiled
    /// kernel to serve arbitrary shapes.
    ///
    /// Resolves any `Unit` axis lane counts using the device `plane_size`. Use
    /// [`launcher_over`](Self::launcher_over) if specific axes should remain static.
    pub fn launcher<R: Runtime>(self, client: &ComputeClient<R>) -> Launcher<'_, R> {
        let plane_size = client.properties().hardware.plane_size_max as usize;
        let concrete = self.resolve_lanes(plane_size);
        let kernel = concrete.clone().all_dynamic();
        Launcher::new(concrete, kernel, client)
    }

    /// Creates a [`Launcher`] where only the specified `dynamic` axes have dynamic extents in
    /// the kernel space; all other axes remain compile-time static.
    ///
    /// Useful to specialize kernel loops along specific axes, or for an axis no operand can state
    /// the size of at runtime ([`Tile::witnesses`](crate::Tile::witnesses)). Passing `&[]` creates
    /// a fully static launch.
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

    /// [`bind`](Self::bind) over a stated geometry rather than a binding: for an
    /// operand with no tensor to bind — the destination a fused store writes
    /// through ([`Tile::of_sink`](crate::Tile::of_sink)), or the producer a fused
    /// read comes from ([`Tile::of_source`](crate::Tile::of_source)).
    ///
    /// `geometry` is the physical extents and strides the operand *would* have
    /// had. Everything else — the projection, the bounds-check derived from this
    /// launcher's concrete overhang, the residence column, the cube size — is
    /// settled exactly as it is for a bound operand, because this is the builder a
    /// bound operand configures: [`batches`](StridedTileSource::batches),
    /// [`vectorize`](StridedTileSource::vectorize),
    /// [`checked`](StridedTileSource::checked),
    /// [`stage_width`](StridedTileSource::stage_width) and
    /// [`tiling`](StridedTileSource::tiling) all read the same here as there, and
    /// an operand that needs one of them tunes it rather than hand-building a spec
    /// beside the derivation.
    ///
    /// End it with [`build_spec`](StridedTileSource::build_spec) rather than
    /// [`build`](StridedTileSource::build): there is no tensor to ship, and the
    /// settled geometry comes back beside the spec, which is what
    /// [`Tile::of_sink`](crate::Tile::of_sink) takes — not the stated one. The two
    /// part company exactly where a broadcast batch dim is dropped, which is why it
    /// is the settled one that travels: reading it keeps the dropping an
    /// implementation detail of the derivation rather than a fact the call site has
    /// to reproduce.
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

    /// The widest `Vector<E, v>` line every operand can be served in along `axis`: one width
    /// for all of them, since a kernel reading one operand's lines writes the other's. Each
    /// `(geometry, subspace)` must be unchecked (no [`overhangs`](Space::overhangs) on its
    /// subspace; a masked access reports its length in lines and would wrongly clip) and
    /// innermost-contiguous; the width must divide each inner extent, every coarser stride, and
    /// the axis's leaf tile edge. `1` (scalar) when nothing wider qualifies.
    ///
    /// A [`Geometry`] rather than a binding, so that an operand with no tensor to bind — the
    /// destination of a fused store — constrains the shared width like any other. It is one
    /// width for *all* of them, and one the destination cannot serve is not a width.
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
