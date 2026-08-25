//! `Strategy::Auto` across representative shapes and precisions: the root's own
//! dispatch, whichever architecture is compiled behind it. It belongs to neither
//! `tiled` nor `multi_level`, and stays correct when either is deleted.

use cubecl::{Runtime, TestRuntime, ir::AddressType, zspace::shape};
use cubek_matmul::{definition::MatmulProblem, strategy::Strategy};
use cubek_std::MatrixLayout;

use crate::harness::{
    assert_batch_broadcast, client, f16_elems, f32_elems, f64_elems, passed, rect,
    run_with_strides, square, test_matmul_strategy,
};

#[test]
fn auto_small_f16() {
    test_matmul_strategy(client(), square(16, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_medium_f16() {
    test_matmul_strategy(client(), square(256, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_medium_f32() {
    test_matmul_strategy(client(), square(256, f32_elems()), Strategy::Auto);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_skinny_vecmat() {
    test_matmul_strategy(client(), rect(1, 256, 256, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_skinny_matvec() {
    test_matmul_strategy(client(), rect(256, 1, 256, f16_elems()), Strategy::Auto);
}

#[cfg(feature = "heavy")]
#[test]
fn auto_medium_f64() {
    test_matmul_strategy(client(), square(256, f64_elems()), Strategy::Auto);
}

/// Stride-0 coverage: an `extended`-tier check, like the rest of the broadcast table.
#[cfg(feature = "extended")]
#[test]
fn batch_broadcast_auto() {
    assert_batch_broadcast(Strategy::Auto);
}

/// The exact reported case: lhs `[300,4,256]` strides `[256,0,1]` (M broadcast on
/// lhs) @ rhs `[1,256,256]` strides `[65536,1,256]` (batch broadcast on rhs).
/// The default `Auto` path must compute it correctly (it does, via
/// `into_contiguous`); this guards against a backend regression.
#[cfg(feature = "extended")]
#[test]
fn reported_m_broadcast() {
    use MatrixLayout::{ColMajor, RowMajor};
    let client = TestRuntime::client(&Default::default());
    let mut problem = MatmulProblem::from_parameters(
        4,
        256,
        256,
        shape![300],
        shape![1],
        RowMajor,
        ColMajor,
        RowMajor,
        None,
        None,
        f32_elems(),
        AddressType::U32,
    );
    problem.lhs_strides[1] = 0;
    let outcome = run_with_strides(client, problem, Strategy::Auto);
    assert!(
        passed(&outcome),
        "reported M-broadcast repro gave {outcome:?}"
    );
}
