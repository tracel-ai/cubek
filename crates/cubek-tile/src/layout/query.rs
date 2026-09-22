//! The questions a launch asks of a [`Projection`] once it has the buffer's real extents and
//! strides in hand — its [`Geometry`].
//!
//! A projection holds neither: which dim carries `k` is a fact about the operand, how far `k`
//! steps is a fact about the allocation, and they arrive from different places. So every
//! question here takes both. The ones that exist are the ones the launches actually ask:
//!
//! * [`contiguous`](Projection::contiguous) — which axis the unit-strided dim carries: all that
//!   "row-major" and "col-major" ever meant for a rank-2 operand, a fact of layout *and* buffer.
//! * [`is_addressable`](Projection::is_addressable) — whether two positions can land on one cell.

use cubecl::zspace::SmallVec;

use crate::{Axis, Composition, Geometry, Projection, Space};

impl Projection {
    /// The axes carried by the buffer's unit-strided dim, or none where no dim strides by one.
    ///
    /// Several come back from a windowed dim, which is the honest answer: a convolution's
    /// innermost spatial dim is addressed by an output step and a tap together.
    pub fn contiguous(&self, geometry: &Geometry) -> SmallVec<[Axis; Space::MAX_RANK]> {
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

    /// Whether the buffer's dims can be addressed without two positions landing on one cell.
    ///
    /// Ordered by stride, each dim must step by at least the span of everything finer than it.
    /// Equality is a dense buffer and a larger stride is padding; only a *shorter* one aliases. An
    /// [`Overlapping`](Composition::Overlapping) projection is exempt: it lands twice by design.
    pub fn is_addressable(&self, geometry: &Geometry) -> bool {
        if self.composition() == Composition::Overlapping {
            return true;
        }
        by_stride(geometry).windows(2).all(|pair| {
            let (_, (_, stride)) = pair[0];
            let (_, (finer_extent, finer_stride)) = pair[1];
            stride >= finer_extent * finer_stride
        })
    }

    /// The buffer's contiguous dim — the innermost unit-strided one.
    fn unit_dim(&self, geometry: &Geometry) -> Option<usize> {
        by_stride(geometry)
            .into_iter()
            .filter(|&(_, (_, stride))| stride == 1)
            .map(|(dim, _)| dim)
            .next_back()
    }
}

/// The dims that actually address something, as `(index, (extent, stride))`, in the buffer's
/// own order.
///
/// A dim of extent one reaches exactly one cell whatever its stride, and a stride-zero dim is a
/// broadcast that aliases on purpose; neither takes part in the orderings the checks rest on.
fn addressing_dims(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> {
    geometry
        .dims()
        .enumerate()
        .filter(|&(_, (extent, stride))| extent > 1 && stride > 0)
        .collect()
}

/// The addressing dims coarsest stride first. Ordering by stride rather than by position is
/// what makes a transposed view and its physical self answer the same.
fn by_stride(geometry: &Geometry) -> SmallVec<[(usize, (usize, usize)); Space::MAX_RANK]> {
    let mut dims = addressing_dims(geometry);
    dims.sort_by_key(|&(_, (_, stride))| core::cmp::Reverse(stride));
    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);
    const B: Axis = Axis(2);
    const H: Axis = Axis(3);
    const S: Axis = Axis(4);
    const D: Axis = Axis(5);
    const M: Axis = Axis(6);
    const C: Axis = Axis(7);

    fn geometry(dims: &[(usize, usize)]) -> Geometry {
        Geometry::new(dims)
    }

    /// A `[k, n]` view over an `[n, k]` buffer: `k` strides by one, `n` by the row. The whole
    /// of what a `Col` variant used to say, read off the strides instead.
    #[test]
    fn a_col_weight_is_k_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 1), (11008, 4096)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[K]);
    }

    #[test]
    fn a_row_weight_is_n_contiguous() {
        let p = Projection::direct(&[K, N]);
        let g = geometry(&[(4096, 11008), (11008, 1)]);

        assert_eq!(p.contiguous(&g).as_slice(), &[N]);
    }

    /// A pitched allocator makes a row longer than its extent: addressable, and every kernel
    /// walks by the stride rather than the shape to serve it.
    #[test]
    fn padding_is_addressable() {
        let p = Projection::direct(&[K, N]);
        let padded = geometry(&[(4096, 11136), (11008, 1)]);

        assert!(p.is_addressable(&padded));
    }

    /// A stride *shorter* than the row is a narrowed or permuted view, not padding.
    #[test]
    fn a_narrowed_row_aliases() {
        let p = Projection::direct(&[K, N]);
        let narrowed = geometry(&[(4096, 4096), (11008, 1)]);

        assert!(!p.is_addressable(&narrowed));
    }

    /// A capacity-shaped cache, padded by the allocator and written in place: padding is
    /// addressable, and so is a permutation, because a permutation is a bijection and aliases
    /// nothing.
    #[test]
    fn an_in_place_cache_is_addressable_padded_or_permuted() {
        let p = Projection::direct(&[B, H, S, D]);

        let padded = geometry(&[(2, 8 * 4096 * 136), (8, 4096 * 136), (4096, 136), (128, 1)]);
        assert!(p.is_addressable(&padded));

        let permuted = geometry(&[(2, 8 * 4096 * 128), (8, 128), (4096, 8 * 128), (128, 1)]);
        assert!(p.is_addressable(&permuted));
    }

    /// A windowed dim is addressed by two axes, and it is not an aliasing fault that
    /// consecutive windows overlap — that is what a receptive field is.
    #[test]
    fn a_window_is_exempt_from_the_aliasing_check() {
        let p = Projection::dims()
            .dim(B)
            .dim(crate::stencil(&[(M, 2), (K, 1)]).pad(1))
            .dim(C)
            .build();
        let g = geometry(&[(8, 64 * 32), (64, 32), (32, 1)]);

        assert_eq!(p.composition(), Composition::Overlapping);
        assert!(p.is_addressable(&g));
        assert_eq!(p.contiguous(&g).as_slice(), &[C]);
    }
}
