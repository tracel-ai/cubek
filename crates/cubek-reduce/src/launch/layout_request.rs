//! Reduce's layout intent in the shared [`LayoutRequest`] vocabulary, with the mapping from a
//! delivered layout to a [`ConcreteLayout`]. Reduce prefers a non-reduce axis innermost (the
//! Perpendicular path: independent lanes), falling back to Parallel (vectorize the reduction)
//! when the reduce axis is the contiguous one.

use cubek_layout::{Axis, AxisSet, ConcreteLayout, Constraint, Facet, LayoutRequest, PhysicalAxis};

/// Reduce wishes a non-reduce axis to land innermost so it can vectorize independent lanes
/// (Perpendicular). Preferred, not required: when the reduce axis is contiguous instead, the
/// kernel still runs (Parallel, a cross-lane reduction).
#[allow(dead_code)]
pub(crate) fn reduce_layout_request(rank: usize, reduce_axis: usize) -> LayoutRequest {
    let non_reduce: Vec<Axis> = (0..rank)
        .filter(|&i| i != reduce_axis)
        .map(|i| Axis(i as u8))
        .collect();
    LayoutRequest::new().with(Constraint::preferred(Facet::Innermost(AxisSet::new(
        &non_reduce,
    ))))
}

/// The delivered layout as a [`ConcreteLayout`]: one axis per tensor dim (`Axis(i)`), ordered
/// major-to-minor by stride, untiled. The innermost (smallest-stride) axis is the contiguous
/// one a vectorized read lands on.
#[allow(dead_code)]
pub(crate) fn concrete_from_strides(shape: &[usize], strides: &[usize]) -> ConcreteLayout {
    let mut order: Vec<usize> = (0..shape.len()).collect();
    order.sort_by(|&a, &b| strides[b].cmp(&strides[a])); // major (large stride) first
    let axes: Vec<PhysicalAxis> = order
        .into_iter()
        .map(|i| PhysicalAxis::new(Axis(i as u8), shape[i]))
        .collect();
    ConcreteLayout::new(&axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorizationMode;

    /// What the live launch picks today: Parallel iff the reduce axis is contiguous.
    fn current_mode(strides: &[usize], reduce_axis: usize) -> VectorizationMode {
        match strides[reduce_axis] {
            1 => VectorizationMode::Parallel,
            _ => VectorizationMode::Perpendicular,
        }
    }

    /// What the request implies: Perpendicular when a non-reduce axis is innermost (the
    /// preferred wish is met), else Parallel.
    fn request_mode(shape: &[usize], strides: &[usize], reduce_axis: usize) -> VectorizationMode {
        let req = reduce_layout_request(shape.len(), reduce_axis);
        let concrete = concrete_from_strides(shape, strides);
        if req.preference(&concrete) > 0 {
            VectorizationMode::Perpendicular
        } else {
            VectorizationMode::Parallel
        }
    }

    fn assert_agrees(shape: &[usize], strides: &[usize], reduce_axis: usize) {
        assert_eq!(
            current_mode(strides, reduce_axis),
            request_mode(shape, strides, reduce_axis),
            "shape={shape:?} strides={strides:?} reduce_axis={reduce_axis}"
        );
    }

    #[test]
    fn reduce_axis_contiguous_is_parallel() {
        // Row-major [4, 8], strides [8, 1]: reducing the contiguous axis 1 -> Parallel.
        assert_agrees(&[4, 8], &[8, 1], 1);
    }

    #[test]
    fn reduce_axis_strided_is_perpendicular() {
        // Same layout, reducing axis 0 (stride 8): axis 1 is innermost -> Perpendicular.
        assert_agrees(&[4, 8], &[8, 1], 0);
    }

    #[test]
    fn rank3_matches_each_axis() {
        // Contiguous [2, 3, 4], strides [12, 4, 1].
        let (shape, strides) = ([2, 3, 4], [12, 4, 1]);
        assert_agrees(&shape, &strides, 0); // reduce outer -> Perpendicular
        assert_agrees(&shape, &strides, 1); // reduce middle -> Perpendicular
        assert_agrees(&shape, &strides, 2); // reduce contiguous -> Parallel
    }

    #[test]
    fn permuted_innermost_axis_drives_it() {
        // Permuted so axis 0 is contiguous (strides [1, 4]); reducing axis 0 -> Parallel,
        // reducing axis 1 -> Perpendicular.
        assert_agrees(&[4, 4], &[1, 4], 0);
        assert_agrees(&[4, 4], &[1, 4], 1);
    }
}
