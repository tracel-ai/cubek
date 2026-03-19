use crate::suite::assert_result;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubecl::{frontend::CubePrimitive, ir::AddressType};
use cubecl::{prelude::TensorBinding, zspace::shape};
use cubek_matmul::launch::launch_vec2mat;
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::vec2mat::{Vec2MatRoutine, Vec2MatStrategy};

use crate::suite::launcher::{InputRepresentation, test_matmul_algorithm};
use crate::suite::layout_to_stride_spec;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_matmul::definition::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_matmul::launch::MatmulInputBinding;
use cubek_std::MatrixLayout;
use cubek_test_utils::{BaseInputSpec, DataKind, Distribution, TestInput};

type TestRuntime = cubecl::TestRuntime;

struct Vec2MatTestCase {
    pub n: usize,
    pub k: usize,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
}

impl Vec2MatTestCase {
    fn into_problem(self) -> MatmulProblem {
        MatmulProblem::from_parameters(
            1,
            self.n,
            self.k,
            shape![1],
            shape![1],
            MatrixLayout::RowMajor,
            self.rhs_layout,
            MatrixLayout::RowMajor,
            None,
            None,
            self.elems,
            AddressType::U32,
        )
    }
}

#[test]
pub fn test_very_small_row_major() {
    let case = Vec2MatTestCase {
        n: 4,
        k: 4 * 32,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_very_small_col_major() {
    let case = Vec2MatTestCase {
        n: 4,
        k: 4,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_small() {
    let case = Vec2MatTestCase {
        n: 64,
        k: 64,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_odd() {
    let case = Vec2MatTestCase {
        n: 255,
        k: 101,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_large() {
    let case = Vec2MatTestCase {
        n: 256,
        k: 256,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_with_check_bounds() {
    let case = Vec2MatTestCase {
        n: 60,
        k: 60,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_with_batches() {
    let case = Vec2MatTestCase {
        n: 64,
        k: 64,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

fn test_vec2mat(case: Vec2MatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let problem = case.into_problem();

    test_matmul_algorithm::<Vec2MatRoutine>(
        client,
        problem,
        BlueprintStrategy::Inferred(Vec2MatStrategy {}),
        InputRepresentation::Normal,
    );

    // let (lhs, lhs_data) = TestInput::new(
    //     client.clone(),
    //     problem.lhs_shape.clone(),
    //     problem.global_dtypes.lhs,
    //     layout_to_stride_spec(problem.lhs_layout),
    //     DataKind::Random {
    //         seed: 1234,
    //         distribution: Distribution::Uniform(-1., 1.),
    //     },
    // )
    // .generate_with_f32_host_data();

    // let (rhs, rhs_data) = TestInput::new(
    //     client.clone(),
    //     problem.rhs_shape.clone(),
    //     problem.global_dtypes.rhs,
    //     layout_to_stride_spec(problem.rhs_layout),
    //     DataKind::Random {
    //         seed: 5678,
    //         distribution: Distribution::Uniform(-1., 1.),
    //     },
    // )
    // .generate_with_f32_host_data();

    // let out = TestInput::new(
    //     client.clone(),
    //     problem.out_shape.clone(),
    //     problem.global_dtypes.out,
    //     layout_to_stride_spec(MatrixLayout::RowMajor),
    //     DataKind::Zeros,
    // )
    // .generate_without_host_data();

    // let lhs_handle = MatmulInputBinding::Normal(lhs.binding(), problem.global_dtypes.lhs);
    // let rhs_handle = MatmulInputBinding::Normal(rhs.binding(), problem.global_dtypes.rhs);
    // let out_handle = out.clone().binding();

    // let all_elems = MatmulElems::from_globals(&problem.global_dtypes.clone());

    // launch_vec2mat::launch_ref(&client, lhs_handle, rhs_handle, out_handle, &all_elems).unwrap();

    // assert_result(&lhs_data, &rhs_data, &problem, &client, out, all_elems);
}
