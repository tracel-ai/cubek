use super::*;
use crate::Axis;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B0: Axis = Axis(3);
const B1: Axis = Axis(4);

/// A plain (untiled) matrix layout whose physical order (major-to-minor) is `order`.
fn matrix(order: &[Axis], extents: &[(Axis, usize)]) -> ConcreteLayout {
    let axes = order
        .iter()
        .map(|&a| {
            let extent = extents.iter().find(|(x, _)| *x == a).unwrap().1;
            PhysicalAxis::untiled(a, extent)
        })
        .collect::<Vec<_>>();
    ConcreteLayout::new(&axes)
}

#[test]
fn empty_request_runs_on_anything() {
    let req = LayoutRequest::new();
    let layout = matrix(&[M, N], &[(M, 16), (N, 16)]);
    assert!(req.feasible(&layout));
    assert_eq!(req.preference(&layout), 0);
}

#[test]
fn innermost_any_of_accepts_row_and_col_major() {
    // The matmul lhs accepts row- or col-major: one of {M, K} innermost.
    let req = LayoutRequest::new().with(Constraint::required(Facet::Innermost(AxisSet::new(&[
        M, K,
    ]))));
    let row_major = matrix(&[M, K], &[(M, 16), (K, 16)]); // K innermost
    let col_major = matrix(&[K, M], &[(M, 16), (K, 16)]); // M innermost
    let n_major = matrix(&[M, N], &[(M, 16), (N, 16)]); // neither innermost
    assert!(req.feasible(&row_major));
    assert!(req.feasible(&col_major));
    assert!(!req.feasible(&n_major));
}

#[test]
fn preferred_ranks_within_feasible() {
    // Feasible on any contiguity, but prefers N innermost (vectorization landing).
    let req = LayoutRequest::new().with(Constraint::preferred(Facet::Innermost(AxisSet::one(N))));
    let n_inner = matrix(&[M, N], &[(M, 16), (N, 16)]);
    let m_inner = matrix(&[N, M], &[(M, 16), (N, 16)]);
    assert!(req.feasible(&n_inner) && req.feasible(&m_inner));
    assert_eq!(req.preference(&n_inner), 1);
    assert_eq!(req.preference(&m_inner), 0);
}

#[test]
fn minor_requires_trailing_axes() {
    // Pool wants the channel/batch axes trailing (innermost slots).
    let req =
        LayoutRequest::new().with(Constraint::required(Facet::Minor(AxisSet::new(&[B0, B1]))));
    let trailing = matrix(&[M, B0, B1], &[(M, 8), (B0, 4), (B1, 4)]);
    let not_trailing = matrix(&[B0, M, B1], &[(M, 8), (B0, 4), (B1, 4)]);
    assert!(req.feasible(&trailing));
    assert!(!req.feasible(&not_trailing));
}

#[test]
fn tiled_is_satisfied_by_a_finer_multiple() {
    let req = LayoutRequest::new().with(Constraint::required(Facet::Tiled { axis: M, edge: 8 }));
    let tiled_8 =
        ConcreteLayout::new(&[PhysicalAxis::tiled(M, 64, 8), PhysicalAxis::untiled(K, 64)]);
    let tiled_16 =
        ConcreteLayout::new(&[PhysicalAxis::tiled(M, 64, 16), PhysicalAxis::untiled(K, 64)]);
    let tiled_12 =
        ConcreteLayout::new(&[PhysicalAxis::tiled(M, 64, 12), PhysicalAxis::untiled(K, 64)]);
    let untiled =
        ConcreteLayout::new(&[PhysicalAxis::untiled(M, 64), PhysicalAxis::untiled(K, 64)]);
    assert!(req.feasible(&tiled_8));
    assert!(req.feasible(&tiled_16)); // finer multiple satisfies
    assert!(!req.feasible(&tiled_12)); // not a multiple
    assert!(!req.feasible(&untiled));
}

#[test]
fn divisible_checks_extent() {
    let req = LayoutRequest::new().with(Constraint::required(Facet::Divisible { axis: N, by: 4 }));
    let aligned = matrix(&[M, N], &[(M, 16), (N, 16)]);
    let ragged = matrix(&[M, N], &[(M, 16), (N, 13)]);
    assert!(req.feasible(&aligned));
    assert!(!req.feasible(&ragged));
}
