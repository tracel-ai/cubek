//! `Strategy::Auto` across representative shapes and precisions: the root's own
//! dispatch, whichever architecture is compiled behind it. It belongs to neither
//! `tiled` nor `multi_level`, and stays correct when either is deleted.

use cubecl::{ir::AddressType, zspace::shape};
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
    let client = cubecl::test_device().client();
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

/// A storage-tiled operand is a fact of the data: `Auto` sends it to the tiled cmma routine,
/// which stages to its tiles, whichever architecture it would have picked for the shape.
#[cfg(feature = "tiled")]
#[test]
fn auto_sends_a_packed_weight_to_the_tiled_cmma() {
    use cubecl::prelude::*;
    use cubek_matmul::{
        definition::{MatmulElems, MatmulSetupError},
        launch::launch_ref,
        tiled::pack::pack,
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};

    use crate::harness::assert_result;

    let client = client();
    let dtype = f32::elem_type_native();
    let dtypes = MatmulElems::from_single_dtype(dtype);
    let problem = rect(64, 512, 256, dtypes.as_global_elems());
    let (lhs, lhs_data) = TestInput::builder(client.clone(), problem.lhs_shape.clone())
        .dtype(dtype)
        .uniform(1234, -1., 1.)
        .generate_with_f32_host_data();
    let (rhs, rhs_data) = TestInput::builder(client.clone(), problem.rhs_shape.clone())
        .dtype(dtype)
        .uniform(5678, -1., 1.)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), problem.out_shape.clone())
        .dtype(dtype)
        .uniform(4242, 10., 100.)
        .generate_without_host_data();
    let packed = pack(&client, rhs.binding(), dtype, (32, 64)).unwrap();

    let mut elems = dtypes.clone();
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        let mut launch = || -> Result<(), MatmulSetupError> {
            launch_ref(
                &Strategy::Auto,
                c,
                InputBinding::Normal(lhs.clone().binding(), dtype),
                InputBinding::Normal(packed.clone().binding(), dtype),
                out.clone().binding(),
                &mut elems,
            )
        };
        launch().into()
    });
    match outcome {
        ExecutionOutcome::Executed => {
            assert_result(&lhs_data, &rhs_data, &problem, &client, out, dtypes).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce()
}
