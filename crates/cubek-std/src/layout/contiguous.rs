use super::fragments::fragments;
use cubecl::zspace::metadata::Metadata;
use cubek_tile::Axis;

/// The one reading of which axis an operand is contiguous along: the axis whose fragments hold the
/// unit-strided dim, over the trailing two logical dims named `axes`, or `None` where no fragment
/// of extent past one strides by one — a strided or broadcast view, which the kernels stage or
/// decline rather than walk in lines. A fragment of one addresses nothing and is not counted, so
/// a `[m, 1]` operand at strides `[1, 1]` is contiguous along `M`.
///
/// Read off the operand's [`fragments`], so a plain buffer, one fragment a dim, and a buffer
/// stored in tiles, several, answer the same way: a `[k, n]` weight stored in `8 × 8` tiles a row
/// at a time is contiguous along `N`.
///
/// Every consumer of the fact (the key, the order the plane shuffle reads a weight in, the axis a
/// packed operand's words run along) reads it through this, so no two of them can disagree about
/// the same layout. The batch dims are left out on purpose: they broadcast, and a leading stride
/// says nothing about which way the product's own dims run.
pub fn contiguous(meta: &Metadata, axes: [Axis; 2]) -> Option<Axis> {
    let dims = fragments(meta);
    let mut lined = axes
        .into_iter()
        .zip(&dims[dims.len() - 2..])
        .filter(|(_, dim)| {
            dim.iter()
                .any(|&(extent, stride)| extent > 1 && stride == 1)
        })
        .map(|(axis, _)| axis);
    match (lined.next(), lined.next()) {
        (Some(axis), None) => Some(axis),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul_axes::{K, M, N};
    use cubecl::zspace::{Shape, Strides, Tiling};

    /// A plain buffer of `shape` at `strides`.
    fn plain(shape: &[usize], strides: &[usize]) -> Metadata {
        Metadata::new(shape.to_vec(), strides.to_vec())
    }

    /// The production weight and its artifact form are the same shape under
    /// opposite strides, and the key tells them apart by the axis each is
    /// contiguous along, which decides the order the plane shuffle reads it in,
    /// so a record shared between them would hand one the other's
    /// winner.
    #[test]
    fn a_weight_keys_by_the_axis_it_is_contiguous_along() {
        let (k, n) = (4096, 6144);
        // `[n, k]` behind a `[k, n]` view: `k` strides by one.
        assert_eq!(contiguous(&plain(&[k, n], &[1, k]), [K, N]), Some(K));
        // `[k, n]` row-major: `n` strides by one.
        assert_eq!(contiguous(&plain(&[k, n], &[n, 1]), [K, N]), Some(N));
        // Padded by the pitched allocator: still `n`, the pad is not a dim.
        assert_eq!(contiguous(&plain(&[k, n], &[n + 128, 1]), [K, N]), Some(N));
    }

    /// The backward's left-hand side arrives transposed, which the engine
    /// serves by its strides; a strided slice is neither orientation and is
    /// staged. Both differ from the row-major product they share a shape with.
    #[test]
    fn a_left_hand_side_keys_transposed_and_strided_apart() {
        let (m, k) = (64, 4096);
        assert_eq!(contiguous(&plain(&[m, k], &[k, 1]), [M, K]), Some(K));
        assert_eq!(contiguous(&plain(&[m, k], &[1, m]), [M, K]), Some(M));
        assert_eq!(contiguous(&plain(&[m, k], &[2 * k, 2]), [M, K]), None);
        // A degenerate extent strides by one too, and does not count: a dim
        // of one addresses nothing, so the other axis is the contiguous one.
        assert_eq!(contiguous(&plain(&[m, 1], &[1, 1]), [M, K]), Some(M));
    }

    /// Only the trailing two dims are read: a batch dim's stride, broadcast
    /// or not, is not the product's orientation.
    #[test]
    fn the_batch_dims_do_not_reach_the_key() {
        assert_eq!(
            contiguous(&plain(&[8, 64, 4096], &[0, 4096, 1]), [M, K]),
            Some(K)
        );
    }

    /// A weight stored in tiles reads the same way as one in rows: `[k, n]` in `8 × 8` tiles, the
    /// tiles in rows and each tile a row at a time, is contiguous along `n`, and stored a column
    /// at a time along `k`.
    #[test]
    fn a_weight_in_tiles_is_contiguous_along_its_innermost_fragment() {
        let (k, n) = (64, 32);
        let tiles = |inner: [usize; 2]| {
            Metadata::new(
                Shape::from(vec![k / 8, n / 8, 8, 8]),
                Strides::from(vec![n * 8, 64, inner[0], inner[1]]),
            )
            .with_tiling(Tiling::new(&[2, 2]).expect("two fragments a dim"))
            .expect("the tiling describes the dims")
        };
        assert_eq!(contiguous(&tiles([8, 1]), [K, N]), Some(N));
        assert_eq!(contiguous(&tiles([1, 8]), [K, N]), Some(K));
    }
}
