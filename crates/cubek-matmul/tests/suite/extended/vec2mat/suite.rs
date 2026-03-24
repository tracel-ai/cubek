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
    pub target_vec: usize,
    pub n_tiles: usize,
    pub k_tiles: usize,
    pub lhs_batch: usize,
    pub rhs_batch: usize,
    pub rhs_layout: MatrixLayout,
    pub elems: MatmulGlobalElems,
}

impl Vec2MatTestCase {
    fn into_problem(self, plane_size: usize) -> MatmulProblem {
        let tile_dim = plane_size * self.target_vec;
        MatmulProblem::from_parameters(
            1,
            self.n_tiles * tile_dim,
            self.k_tiles * tile_dim,
            shape![self.lhs_batch],
            shape![self.rhs_batch],
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
pub fn test_very_small_square_rhs_row_major() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 1,
        k_tiles: 1,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_k_larger_than_n() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 1,
        k_tiles: 2,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_k_smaller_than_n() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 2,
        k_tiles: 1,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_small_square_rhs_row_major() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 2,
        k_tiles: 2,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_large() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 10,
        k_tiles: 10,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_large_broadcast_lhs() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 10,
        k_tiles: 10,
        lhs_batch: 1,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_large_broadcast_rhs() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 10,
        k_tiles: 10,
        lhs_batch: 2,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

#[test]
pub fn test_large_broadcast_batched() {
    let case = Vec2MatTestCase {
        target_vec: 4,
        n_tiles: 10,
        k_tiles: 10,
        lhs_batch: 2,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
    };

    test_vec2mat(case);
}

fn test_vec2mat(case: Vec2MatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let problem = case.into_problem(plane_size);

    test_matmul_algorithm::<Vec2MatRoutine>(
        client,
        problem,
        BlueprintStrategy::Inferred(Vec2MatStrategy {
            target_num_planes: 8,
        }),
        InputRepresentation::Normal,
    );
}
