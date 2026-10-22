//! Smoke tests for the tile-DSL `Cmma` strategy: the cube staged routine of
//! [`kernels`](cubek_matmul::kernels), reached through `launch_ref` as burn reaches it.

use cubecl::prelude::*;
use cubek_matmul::{kernels::CubeStagedMatmulRoutineSettings, tiled::Strategy as Tiled};

use crate::harness::{client, f16_elems, f32_elems, rect, square, test_matmul_strategy};

fn cmma() -> Tiled {
    Tiled::Cmma(CubeStagedMatmulRoutineSettings::DEFAULT)
}

#[test]
fn cmma_square_f32() {
    test_matmul_strategy(client(), square(256, f32_elems()), cmma().into());
}

#[test]
fn cmma_square_f16() {
    test_matmul_strategy(client(), square(256, f16_elems()), cmma().into());
}

#[test]
fn cmma_rect_f32() {
    test_matmul_strategy(client(), rect(64, 128, 32, f32_elems()), cmma().into());
}

#[test]
fn cmma_batched_f32() {
    use cubecl::{ir::AddressType, zspace::shape};
    use cubek_matmul::definition::MatmulProblem;
    use cubek_std::MatrixLayout;

    let elems = f32_elems();
    let problem = MatmulProblem::from_parameters(
        64,
        64,
        64,
        shape![3],
        shape![3],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    );
    test_matmul_strategy(client(), problem, cmma().into());
}

/// The weight stored in `32 × 32` tiles beside a row-major activation: the same product as the
/// row-major run, the routine reading each tile through the binding's tiling.
#[test]
fn cmma_storage_tiled_weight_f16() {
    use cubecl::frontend::Scalar;
    storage_tiled_weight(half::f16::elem_type_native());
}

#[test]
fn cmma_storage_tiled_weight_f32() {
    use cubecl::frontend::Scalar;
    storage_tiled_weight(f32::elem_type_native());
}

fn storage_tiled_weight(dtype: ElemType) {
    use cubecl::tile::Axis;
    use cubek_matmul::{
        definition::{MatmulElems, MatmulSetupError},
        launch::launch_ref,
        tiled::storage::{StorageLevels, tile},
    };
    use cubek_std::InputBinding;
    use cubek_test_utils::{ExecutionOutcome, TestInput, TestOutcome, launch_and_capture_outcome};
    const K: Axis = Axis(0);
    const N: Axis = Axis(1);

    use crate::harness::assert_result;

    let client = client();
    let (m, n, k) = (128, 256, 384);
    let dtypes = MatmulElems::from_single_dtype(dtype);
    let problem = rect(m, n, k, dtypes.as_global_elems());

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

    let mut elems = dtypes.clone();
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |c| {
        let mut launch = || -> Result<(), MatmulSetupError> {
            let storage = StorageLevels::new(&[(N, 32), (K, 32)]).grid(&[N, K]);
            let tiled = tile(c, rhs.clone().binding(), &[K, N], dtype, None, storage)?;
            launch_ref(
                &cmma().into(),
                c,
                InputBinding::Normal(lhs.clone().binding(), dtype),
                InputBinding::Normal(tiled.binding(), dtype),
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
    .enforce();
}
