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
        n: 32 * 4,
        k: 32 * 4,
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
        BlueprintStrategy::Inferred(Vec2MatStrategy {
            target_num_planes: 8,
        }),
        InputRepresentation::Normal,
    );
}
