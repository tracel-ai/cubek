//! The [`Launcher`]: a concrete [`Space`] bound to a client for one kernel launch. It keeps
//! the concrete (real-extent) space alongside the derived kernel-form (dynamic) one, so
//! geometry and divisibility are always read off real extents and no call site can consume
//! the space too early.

use cubecl::prelude::*;

use crate::{Axis, Set, Space, StridedTileArgLaunch, StridedTileSource, Unset};

/// One launch's host-side bundle: the concrete space (real extents, for geometry, overhang and
/// divisibility math) and the kernel-form space tile arguments project from.
pub struct Launcher<'c, R: Runtime> {
    concrete: Space,
    kernel: Space,
    client: &'c ComputeClient<R>,
}

impl Space {
    /// Bind this concrete (real-extent) space to `client` for launching. The kernel-form space
    /// goes fully dynamic so one compiled kernel serves every shape; a static-shape constructor
    /// can later skip that derivation without changing this one.
    pub fn launcher<R: Runtime>(self, client: &ComputeClient<R>) -> Launcher<'_, R> {
        let kernel = self.clone().all_dynamic();
        Launcher {
            concrete: self,
            kernel,
            client,
        }
    }
}

impl<'c, R: Runtime> Launcher<'c, R> {
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

    /// Start a tile argument over the kernel space: [`StridedTileArgLaunch::source`] with
    /// [`space`](StridedTileSource::space) pre-set and the bounds-check derived from the concrete
    /// space's overhang (an explicit [`checked`](StridedTileSource::checked) still wins).
    pub fn arg<E: Numeric>(
        &self,
        binding: TensorBinding<R>,
    ) -> StridedTileSource<'_, Set, Unset, E, R> {
        StridedTileArgLaunch::source(binding)
            .space(&self.kernel)
            .concrete(&self.concrete)
            .cube_units(self.cube_dim().num_elems() as usize)
    }

    /// The widest `Vector<E, v>` line every operand can be served in along `axis` — one width
    /// for all of them, since a kernel reading one operand's lines writes the other's. Each
    /// `(binding, subspace)` must be unchecked (no [`overhangs`](Space::overhangs) on its
    /// subspace — a masked access reports its length in lines and would wrongly clip) and
    /// innermost-contiguous; the width must divide each inner buffer extent, every coarser
    /// stride, and the axis's leaf tile edge. `1` (scalar) when nothing wider qualifies.
    pub fn vector_size(
        &self,
        axis: Axis,
        operands: &[(&TensorBinding<R>, &[Axis])],
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
        let qualifies = operands.iter().all(|(binding, subspace)| {
            binding.strides.last() == Some(&1)
                && !subspace.iter().any(|&a| self.concrete.overhangs(a))
        });
        if !qualifies {
            return 1;
        }
        let leaf = self.concrete.final_space().extent(axis);
        self.client
            .io_optimized_vector_sizes(type_size)
            .filter(|&v| {
                leaf.is_multiple_of(v)
                    && operands.iter().all(|(b, _)| {
                        b.shape.last().is_some_and(|&e| e.is_multiple_of(v))
                            // Coarser strides re-express in lines (`stride / v`), so `v`
                            // must divide them or a padded/sliced view truncates.
                            && b.strides[..b.strides.len() - 1]
                                .iter()
                                .all(|&s| s.is_multiple_of(v))
                    })
            })
            .max()
            .unwrap_or(1)
    }
}
