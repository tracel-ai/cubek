//! The questions a launch asks of a [`Projection`] once it has the buffer's real extents and
//! strides in hand — its [`Geometry`].
//!
//! A projection holds neither: which dim carries `k` is a fact about the operand, how far `k`
//! steps is a fact about the allocation, and they arrive from different places. So every
//! question here takes both. The ones that exist are the ones the launches actually ask:
//!
//! * [`contiguous`](Projection::contiguous) — which axis the unit-strided dim carries. This is
//!   all that "row-major" and "col-major" ever meant for a rank-2 operand, and it is a question
//!   about the layout *and* the buffer, never about the layout alone.
//! * [`stride_of`](Projection::stride_of) — how far an axis steps, off the allocation rather
//!   than the shape, because a pitched allocator pads a row past its extent.
//! * [`is_addressable`](Projection::is_addressable) — whether two positions can land on one
//!   cell. The question a launch asks before deciding to copy an operand.
//! * [`is_ordered`](Projection::is_ordered) — whether the dims run in the order they are named.
//!   A permuted buffer is a bijection, so it passes the last test and fails this one; a kernel
//!   that walks strides positionally needs this, not that.
//! * [`in_buffer_order`](Projection::in_buffer_order) — the same buffer with its dims named in
//!   the order they actually step, which is what re-expressing a transposed view is.
//! * [`verify`](Projection::verify) — all of the above that must hold, plus that every axis
//!   multiplies back to its extent, as one check with one error.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::SmallVec;

use crate::{Axis, Composition, Geometry, MAX_AXES, PhysicalAxisMap, Projection};

/// Why a [`Projection`] does not describe the buffer it was checked against. Carries the value
/// that decided it, so a message names the number a reader would otherwise go looking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMisfit {
    /// The buffer has a different number of dims than the projection addresses.
    Rank { expected: usize, actual: usize },
    /// A disjoint projection whose dims alias: some dim's stride is shorter than the span of
    /// everything finer than it, so two coordinates address one cell. Padding — a stride
    /// *longer* than that span — is fine and expected.
    Aliases {
        dim: usize,
        stride: usize,
        span: usize,
    },
    /// An axis whose dims do not multiply back to its extent — a storage-tiled axis whose
    /// fragments were misstated, or a shape that is not this problem's.
    Extent {
        axis: Axis,
        expected: usize,
        actual: usize,
    },
}

impl Display for ProjectionMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rank { expected, actual } => write!(
                f,
                "the projection addresses {expected} dims, the buffer has {actual}"
            ),
            Self::Aliases { dim, stride, span } => write!(
                f,
                "dim {dim} strides by {stride}, shorter than the {span} its finer dims span, so \
                 two positions would address one cell"
            ),
            Self::Extent {
                axis,
                expected,
                actual,
            } => write!(
                f,
                "{axis:?} has extent {expected}, but its dims multiply to {actual}"
            ),
        }
    }
}

impl Projection {
    /// The axes carried by the buffer's unit-strided dim, or none where no dim strides by one.
    ///
    /// Several come back from a windowed dim, which is the honest answer: a convolution's
    /// innermost spatial dim is addressed by an output step and a tap together.
    pub fn contiguous(&self, geometry: &Geometry) -> SmallVec<[Axis; MAX_AXES]> {
        match self.unit_dim(geometry) {
            None => SmallVec::new(),
            Some(dim) => self
                .logical_axes()
                .iter()
                .copied()
                .filter(|&axis| self.physical_axis(dim).addresses(axis))
                .collect(),
        }
    }

    /// The stride `axis` steps by: its coarsest dim's, in scalars, off the allocation.
    ///
    /// # Panics
    ///
    /// If this operand does not address `axis` — a spanning axis has no stride, and asking for
    /// one is a staging bug rather than a layout to serve.
    pub fn stride_of(&self, axis: Axis, geometry: &Geometry) -> usize {
        let dim = (0..self.physical_rank())
            .find(|&dim| self.physical_axis(dim).addresses(axis))
            .unwrap_or_else(|| panic!("Projection::stride_of: {axis:?} addresses no dim"));
        geometry.strides()[dim]
    }

    /// Whether the buffer's dims can be addressed without two positions landing on one cell.
    ///
    /// Ordered by stride, each dim must step by at least the span of everything finer than it.
    /// Equality is a dense buffer and a larger stride is padding; only a *shorter* one aliases.
    /// An [`Overlapping`](Composition::Overlapping) projection is exempt, because landing twice
    /// on a cell is what it is for.
    pub fn is_addressable(&self, geometry: &Geometry) -> bool {
        self.alias_check(geometry).is_ok()
    }

    /// Whether the buffer's dims run in the order the projection names them — strides
    /// non-increasing from the first dim to the last.
    ///
    /// A **separate** question from [`is_addressable`](Self::is_addressable): a permutation is
    /// a bijection, so nothing aliases, but a kernel that walks `strides[0..rank]` positionally
    /// reads the wrong axis. Being unordered is not a fault — a `[k, n]` view over an `[n, k]`
    /// buffer is deliberately unordered, and [`in_buffer_order`](Self::in_buffer_order) is how
    /// it is read as its physical self.
    pub fn is_ordered(&self, geometry: &Geometry) -> bool {
        let dims = addressing_dims(geometry);
        dims.windows(2).all(|pair| {
            let (_, (_, stride)) = pair[0];
            let (_, (_, finer)) = pair[1];
            stride >= finer
        })
    }

    /// This projection and buffer with the dims named in the order they actually step,
    /// coarsest stride first. The same bytes and never a copy.
    ///
    /// The logical axes are untouched, because a view changes where the bytes are read, not
    /// what the operand spans. Both halves move together and come back as a pair, so the map
    /// and the strides cannot be permuted by hand and drift apart.
    pub fn in_buffer_order(&self, geometry: &Geometry) -> (Projection, Geometry) {
        let mut order: SmallVec<[usize; MAX_AXES]> = (0..self.physical_rank()).collect();
        order.sort_by_key(|&dim| core::cmp::Reverse(geometry.strides()[dim]));
        let maps: SmallVec<[PhysicalAxisMap; MAX_AXES]> = order
            .iter()
            .map(|&dim| self.physical_axis(dim).clone())
            .collect();
        let dims: SmallVec<[(usize, usize); MAX_AXES]> = order
            .iter()
            .map(|&dim| (geometry.shape()[dim], geometry.strides()[dim]))
            .collect();
        (
            Projection::new(self.logical_axes(), &maps),
            Geometry::of_dims(&dims),
        )
    }

    /// Check this projection against a real buffer and the operand's extents.
    ///
    /// Three things: the ranks agree, a disjoint projection does not alias, and every axis whose
    /// dims carry it alone multiplies back to its extent — which is what catches a storage-tiled
    /// axis whose fragments were misstated. A shared dim is exempt from the last, since a
    /// window's spatial extent is not the product of its output and tap extents.
    pub fn verify(
        &self,
        geometry: &Geometry,
        extent_of: impl Fn(Axis) -> usize,
    ) -> Result<(), ProjectionMisfit> {
        if geometry.rank() != self.physical_rank() {
            return Err(ProjectionMisfit::Rank {
                expected: self.physical_rank(),
                actual: geometry.rank(),
            });
        }
        self.alias_check(geometry)?;
        for &axis in self.logical_axes() {
            let carriers: SmallVec<[usize; MAX_AXES]> = (0..self.physical_rank())
                .filter(|&dim| self.physical_axis(dim).addresses(axis))
                .collect();
            // Only where every carrier carries this axis alone: a shared dim's extent belongs
            // to the split or the window, not to one axis.
            let exclusive = carriers
                .iter()
                .all(|&dim| self.physical_axis(dim).terms().len() == 1);
            if carriers.is_empty() || !exclusive {
                continue;
            }
            let actual: usize = carriers.iter().map(|&dim| geometry.shape()[dim]).product();
            let expected = extent_of(axis);
            if actual != expected {
                return Err(ProjectionMisfit::Extent {
                    axis,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// The buffer's contiguous dim — the innermost unit-strided one.
    fn unit_dim(&self, geometry: &Geometry) -> Option<usize> {
        by_stride(geometry)
            .into_iter()
            .filter(|&(_, (_, stride))| stride == 1)
            .map(|(dim, _)| dim)
            .next_back()
    }

    /// The aliasing check both predicates rest on, keeping the dim index so the error can
    /// name it.
    fn alias_check(&self, geometry: &Geometry) -> Result<(), ProjectionMisfit> {
        if self.composition() == Composition::Overlapping {
            return Ok(());
        }
        for pair in by_stride(geometry).windows(2) {
            let (dim, (_, stride)) = pair[0];
            let (_, (finer_extent, finer_stride)) = pair[1];
            let span = finer_extent * finer_stride;
            if stride < span {
                return Err(ProjectionMisfit::Aliases { dim, stride, span });
            }
        }
        Ok(())
    }
}

/// The dims that actually address something, as `(index, (extent, stride))`, in the buffer's
/// own order.
///
/// A dim of extent one reaches exactly one cell whatever its stride, and a stride-zero dim is a
/// broadcast that aliases on purpose; neither takes part in the orderings the checks rest on.
fn addressing_dims(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); MAX_AXES]> {
    geometry
        .dims()
        .enumerate()
        .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
        .collect()
}

/// The addressing dims coarsest stride first. Ordering by stride rather than by position is
/// what makes a transposed view and its physical self answer the same.
fn by_stride(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); MAX_AXES]> {
    let mut dims = addressing_dims(geometry);
    dims.sort_by_key(|&(_, (_, stride))| core::cmp::Reverse(stride));
    dims
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split;

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);
    const B: Axis = Axis(2);
    const H: Axis = Axis(3);
    const S: Axis = Axis(4);
    const D: Axis = Axis(5);
    const M: Axis = Axis(6);
    const C: Axis = Axis(7);
    const G: Axis = Axis(8);
    const T: Axis = Axis(9);

    fn geometry(dims: &[(usize, usize)]) -> Geometry {
        Geometry::of_dims(dims)
    }

    /// A `[k, n]` view over an `[n, k]` buffer: `k` strides by one, `n` by the row. The whole
    /// of what a `Col` variant used to say, read off the strides instead.
    #[test]
    fn a_col_weight_is_k_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 1), (11008, 4096)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[K]);
        assert_eq!(p.stride_of(N, &g), 4096);
        assert_eq!(p.stride_of(K, &g), 1);
        // Declared `[k, n]` over an `[n, k]` buffer: a view, not a fault.
        assert!(!p.is_ordered(&g));
    }

    #[test]
    fn a_row_weight_is_n_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 11008), (11008, 1)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[N]);
        assert_eq!(p.stride_of(K, &g), 11008);
        assert!(p.is_ordered(&g));
    }

    /// A pitched allocator makes a row longer than its extent: addressable, and every kernel
    /// walks by the stride rather than the shape to serve it.
    #[test]
    fn padding_is_addressable() {
        let p = Projection::direct(&[K, N]);
        let padded = geometry(&[(4096, 11136), (11008, 1)]);

        assert!(p.is_addressable(&padded));
        assert!(
            p.verify(&padded, |a| if a == K { 4096 } else { 11008 })
                .is_ok()
        );
    }

    /// A stride *shorter* than the row is a narrowed or permuted view, not padding.
    #[test]
    fn a_narrowed_row_aliases() {
        let p = Projection::direct(&[K, N]);
        let narrowed = geometry(&[(4096, 4096), (11008, 1)]);

        assert!(!p.is_addressable(&narrowed));
        assert_eq!(
            p.verify(&narrowed, |a| if a == K { 4096 } else { 11008 }),
            Err(ProjectionMisfit::Aliases {
                dim: 0,
                stride: 4096,
                span: 11008,
            })
        );
    }

    /// Seven dims over five axes, `m` and `k` each split across two: the tiled axes multiply
    /// back through their fragments.
    #[test]
    fn a_tiled_operand_multiplies_back_to_its_extents() {
        let p = Projection::dims()
            .dim(B)
            .dim(H)
            .dim(M)
            .dim(K)
            .dim(M)
            .dim(K)
            .dim(C)
            .build();
        let g = geometry(&[
            (2, 8 * 4 * 64 * 32 * 128 * 16),
            (8, 4 * 64 * 32 * 128 * 16),
            (4, 64 * 32 * 128 * 16),
            (64, 32 * 128 * 16),
            (32, 128 * 16),
            (128, 16),
            (16, 1),
        ]);
        let extents = |axis: Axis| match axis {
            a if a == B => 2,
            a if a == H => 8,
            a if a == M => 4 * 32,
            a if a == K => 64 * 128,
            a if a == C => 16,
            _ => unreachable!(),
        };

        assert_eq!(p.contiguous(&g).as_slice(), &[C]);
        assert!(p.verify(&g, extents).is_ok());

        // Claim m is 4×32 on a buffer whose fragments are 4×16, and it does not describe it.
        let misstated = geometry(&[
            (2, 8 * 4 * 64 * 16 * 128 * 16),
            (8, 4 * 64 * 16 * 128 * 16),
            (4, 64 * 16 * 128 * 16),
            (64, 16 * 128 * 16),
            (16, 128 * 16),
            (128, 16),
            (16, 1),
        ]);
        assert_eq!(
            p.verify(&misstated, extents),
            Err(ProjectionMisfit::Extent {
                axis: M,
                expected: 128,
                actual: 64,
            })
        );
    }

    /// A split-merge partials buffer: `g` and `t` share a dim, so neither is checked against
    /// its own extent — the dim's extent belongs to the split.
    #[test]
    fn a_shared_dim_is_exempt_from_the_extent_check() {
        let p = Projection::dims()
            .dim(B)
            .dim(split(&[(G, 4), (T, 8)]))
            .dim(D)
            .build();
        let g = geometry(&[(2, 32 * 128), (32, 128), (128, 1)]);
        let extents = |axis: Axis| match axis {
            a if a == B => 2,
            a if a == G => 4,
            a if a == T => 8,
            a if a == D => 128,
            _ => unreachable!(),
        };
        assert!(p.verify(&g, extents).is_ok());
    }

    /// A capacity-shaped cache, padded by the allocator and written in place. Padding is
    /// allowed and a permutation is not — and the two are different predicates, because a
    /// permutation is a bijection and aliases nothing.
    #[test]
    fn an_in_place_cache_distinguishes_padding_from_permutation() {
        let p = Projection::direct(&[B, H, S, D]);

        let padded = geometry(&[(2, 8 * 4096 * 136), (8, 4096 * 136), (4096, 136), (128, 1)]);
        assert!(p.is_addressable(&padded));
        assert!(p.is_ordered(&padded));

        let permuted = geometry(&[(2, 8 * 4096 * 128), (8, 128), (4096, 8 * 128), (128, 1)]);
        assert!(p.is_addressable(&permuted));
        assert!(!p.is_ordered(&permuted));
    }

    /// The swapped binding: the buffer read as its physical self, both halves moving together,
    /// and the same answers because it is the same buffer.
    #[test]
    fn in_buffer_order_moves_the_map_and_the_strides_together() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 1), (11008, 4096)]);

        let (physical, moved) = p.in_buffer_order(&g);

        assert_eq!(moved.shape(), &[11008, 4096]);
        assert_eq!(moved.strides(), &[4096, 1]);
        assert_eq!(physical.contiguous(&moved).as_slice(), &[K]);
        assert_eq!(physical.stride_of(N, &moved), 4096);
        assert_eq!(physical.logical_axes(), p.logical_axes());
        assert!(physical.is_ordered(&moved));

        // Already in buffer order: nothing moves.
        let dense = geometry(&[(4096, 11008), (11008, 1)]);
        assert_eq!(p.in_buffer_order(&dense).1, dense);
    }

    /// A windowed dim is addressed by two axes, and it is not an aliasing fault that
    /// consecutive windows overlap — that is what a receptive field is.
    #[test]
    fn a_window_is_exempt_from_the_aliasing_check() {
        let p = Projection::dims()
            .dim(B)
            .dim(crate::window(&[(M, 2), (K, 1)]).pad(1))
            .dim(C)
            .build();
        let g = geometry(&[(8, 64 * 32), (64, 32), (32, 1)]);

        assert_eq!(p.composition(), Composition::Overlapping);
        assert!(p.is_addressable(&g));
        assert_eq!(p.contiguous(&g).as_slice(), &[C]);
    }
}
