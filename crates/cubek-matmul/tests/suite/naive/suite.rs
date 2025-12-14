use crate::suite::assert_result;
use cubecl::frontend::CubePrimitive;
use cubecl::prelude::TensorHandleRef;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubek_matmul::MatmulInputHandleRef;

use cubek_matmul::components::MatrixLayout;
use cubek_matmul::components::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_matmul::kernels::naive;
use cubek_std::test_utils::SimpleInputSpec;
use cubek_std::test_utils::{Distribution, TestInput};

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
    fn to_problem(self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: vec![self.batch],
            rhs_batches: vec![self.batch],
            out_batches: vec![self.batch],
            lhs_strides: vec![self.m * self.k, self.k],
            rhs_strides: vec![self.k * self.n, self.n],
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
        }
    }
}

#[test]
pub fn test_very_small() {
    let case = MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 3,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
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
pub fn test_very_small_col_major() {
    let case = MatmulTestCase {
        m: 4,
        n: 4,
        k: 4,
        batch: 2,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
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
    let problem = case.to_problem();

    let dtype = elem();
    let lhs_shape = problem.shape(MatmulIdent::Lhs);
    let rhs_shape = problem.shape(MatmulIdent::Rhs);

    let (lhs, lhs_data) = TestInput::random(
        client.clone(),
        lhs_shape.clone(),
        *dtype,
        1234,
        Distribution::Uniform(-1., 1.),
        problem.lhs_layout.into(),
    )
    .generate_with_f32_host_data()
    .unwrap();

    let (rhs, rhs_data) = TestInput::random(
        client.clone(),
        rhs_shape.clone(),
        *dtype,
        5678,
        Distribution::Uniform(-1., 1.),
        problem.rhs_layout.into(),
    )
    .generate_with_f32_host_data()
    .unwrap();

    let out = TestInput::zeros(client.clone(), problem.shape(MatmulIdent::Out), *dtype)
        .generate_without_host_data()
        .unwrap();

    let lhs_handle = MatmulInputHandleRef::Normal(lhs.as_ref(), dtype.dtype);
    let rhs_handle = MatmulInputHandleRef::Normal(rhs.as_ref(), dtype.dtype);
    let out_handle = out.as_ref();

    naive::launch_ref(
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
