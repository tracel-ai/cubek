use crate::suite::assert_result;
use cubecl::frontend::CubePrimitive;
use cubecl::prelude::TensorHandleRef;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubek_matmul::launch::launch_naive;

use crate::suite::layout_to_stride_spec;
use cubek_matmul::definition::MatrixLayout;
use cubek_matmul::definition::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_matmul::launch::MatmulInputHandleRef;
use cubek_matmul::routines::naive;
use cubek_test_utils::{Distribution, SimpleInputSpec, TestInput};

type TestRuntime = cubecl::TestRuntime;

struct MatmulTestCase {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
}

impl MatmulTestCase {
    fn into_problem(self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            self.m,
            self.n,
            self.k,
            vec![self.batch],
            self.lhs_layout,
            self.rhs_layout,
            MatrixLayout::RowMajor,
        )
    }
}

#[test]
pub fn test_very_small() {
    let case = MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_very_small_col_major() {
    let case = MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_small() {
    let case = MatmulTestCase {
        m: 64,
        n: 64,
        k: 64,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_odd() {
    let case = MatmulTestCase {
        m: 1,
        n: 255,
        k: 101,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_large() {
    let case = MatmulTestCase {
        m: 256,
        n: 256,
        k: 256,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_with_check_bounds() {
    let case = MatmulTestCase {
        m: 60,
        n: 60,
        k: 60,
        batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

#[test]
pub fn test_with_batches() {
    let case = MatmulTestCase {
        m: 64,
        n: 64,
        k: 64,
        batch: 3,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };

    test_naive(case);
}

fn test_naive(case: MatmulTestCase) {
    let client = TestRuntime::client(&Default::default());
    let problem = case.into_problem();

    let dtype = elem();

    let (lhs, lhs_data) = TestInput::random(
        client.clone(),
        problem.lhs_shape.clone(),
        *dtype,
        1234,
        Distribution::Uniform(-1., 1.),
        layout_to_stride_spec(problem.lhs_layout),
    )
    .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::random(
        client.clone(),
        problem.rhs_shape.clone(),
        *dtype,
        5678,
        Distribution::Uniform(-1., 1.),
        layout_to_stride_spec(problem.rhs_layout),
    )
    .generate_with_f32_host_data();

    let out = TestInput::zeros(
        client.clone(),
        problem.out_shape.clone(),
        *dtype,
        layout_to_stride_spec(MatrixLayout::RowMajor),
    )
    .generate_without_host_data();

    let lhs_handle = MatmulInputHandleRef::Normal(lhs.as_ref(), dtype.dtype);
    let rhs_handle = MatmulInputHandleRef::Normal(rhs.as_ref(), dtype.dtype);
    let out_handle = out.as_ref();

    launch_naive::launch_ref(
        &client,
        &lhs_handle,
        &rhs_handle,
        &out_handle,
        &MatmulElems::from_single_dtype(dtype),
    )
    .unwrap();

    assert_result(
        &lhs_data,
        &rhs_data,
        &problem,
        &client,
        &out,
        MatmulElems::from_single_dtype(dtype),
    );
}
