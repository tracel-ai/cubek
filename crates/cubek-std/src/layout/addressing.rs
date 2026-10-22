//! How a layout addresses its elements: whether it holds them densely in rows, and whether any two
//! of its indices meet on one element.

/// Whether `shape`/`strides` describe a row-major layout with no gaps, ignoring
/// the strides of extent-1 axes: whether a copy into dense rows would reproduce
/// the buffer.
///
/// Deliberately more permissive than cubecl's `is_contiguous`, which compares
/// every axis against the compacted strides and so calls a dense buffer strided
/// whenever a size-1 axis carries a stride from some earlier shape.
pub fn is_dense(shape: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1;
    // Innermost outward: every axis that addresses more than one element must
    // begin exactly where the axes inside it end. An axis of extent 1 both
    // skips the check (its stride is unreachable) and leaves `expected` alone
    // (it spans one element).
    for (axis, dim) in shape.iter().enumerate().rev() {
        if *dim != 1 && strides[axis] != expected {
            return false;
        }
        expected *= *dim;
    }
    true
}

/// Whether no two indices of `shape`/`strides` address the same element: what
/// makes a buffer safe to write, and to read by strides alone.
///
/// An axis of extent 1 addresses one element whatever its stride, zero
/// included, so only a broadcast axis — a zero stride over more than one
/// index — maps many indices onto one element.
pub fn is_nonoverlapping(shape: &[usize], strides: &[usize]) -> bool {
    if shape
        .iter()
        .zip(strides)
        .any(|(&extent, &stride)| stride == 0 && extent > 1)
    {
        return false;
    }
    let mut dims = shape.iter().zip(strides.iter()).collect::<Vec<_>>();
    dims.sort_by_key(|(_, stride)| **stride);

    // Walking outward from the innermost axis: each axis must begin past
    // the last element the ones inside it reach. An axis of extent 1 is
    // exempt — it addresses nothing of its own, so any stride is free.
    let mut max_offset = 0;
    for (shape, stride) in dims.into_iter() {
        if *stride <= max_offset && *shape != 1 {
            return false;
        }
        max_offset += (*shape - 1) * *stride;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{is_dense, is_nonoverlapping};

    /// A transposed or row-pitched view addresses every element once; a broadcast axis or a view
    /// that folds onto itself does not; an axis of extent 1 is free to carry a zero stride.
    #[test]
    fn only_a_broadcast_or_a_fold_overlaps() {
        assert!(is_nonoverlapping(&[48, 40], &[1, 48]));
        assert!(is_nonoverlapping(&[33, 48], &[56, 1]));
        assert!(is_nonoverlapping(&[1, 48, 40], &[0, 40, 1]));
        assert!(!is_nonoverlapping(&[3, 48, 40], &[0, 40, 1]));
        assert!(!is_nonoverlapping(&[48, 40], &[20, 1]));
    }

    /// The compacted strides are dense, which is the case that was never in
    /// doubt.
    #[test]
    fn compact_strides_are_dense() {
        assert!(is_dense(&[2, 4, 8, 96], &[3072, 768, 96, 1]));
    }

    /// The shape this exists for: a decode activation off `split_heads`, whose
    /// `swap_dims(1, 2)` moved an axis of extent 1. The sequence stride is the
    /// pre-swap one and no index can reach it, so the buffer is still dense —
    /// and cubecl's `is_contiguous` says otherwise, which is the copy.
    #[test]
    fn an_extent_one_axis_may_carry_any_stride() {
        let (heads, head_dim) = (4, 96);
        let (hd, seq_stride) = (heads * head_dim, heads * head_dim);
        assert!(is_dense(
            &[2, heads, 1, head_dim],
            &[hd, head_dim, seq_stride, 1],
        ));
        assert!(!cubecl::std::tensor::is_contiguous(
            &[2, heads, 1, head_dim],
            &[hd, head_dim, seq_stride, 1],
        ));
    }

    /// A real permutation is not dense: swap two axes that both address more
    /// than one element and the elements genuinely are out of order.
    #[test]
    fn a_permutation_of_addressable_axes_is_not_dense() {
        // (2, 4, 7, 96) row-major, with the head and sequence axes swapped.
        assert!(!is_dense(&[2, 4, 7, 96], &[2688, 96, 384, 1]));
    }

    /// A pitched allocation is not dense: the padding after each row is a gap,
    /// and the axis carrying it addresses many elements. This is the case a
    /// relaxed check must still refuse — reading it as dense walks into the
    /// padding on every row.
    #[test]
    fn a_padded_row_is_not_dense() {
        // cubecl's pitched allocator rounds a 96-element f16 row (192 bytes) up
        // to 128 elements.
        assert!(!is_dense(&[1, 2, 8, 96], &[2048, 1024, 128, 1]));
    }

    /// Every axis but one being degenerate is the limit of the exemption: a
    /// single addressable axis with unit stride is dense whatever the rest say.
    #[test]
    fn only_the_addressable_axis_has_to_line_up() {
        assert!(is_dense(&[1, 1, 1, 64], &[999, 64, 777, 1]));
        assert!(!is_dense(&[1, 1, 1, 64], &[999, 64, 777, 2]));
    }
}
