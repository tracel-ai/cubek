use crate::suite::assert_result;
use crate::suite::test_matmul_strategy;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, client};
use cubecl::{frontend::CubePrimitive, ir::AddressType};
use cubecl::{prelude::TensorBinding, zspace::shape};
use cubek_matmul::launch::{Strategy, launch_vec2mat};
use cubek_matmul::routines::BlueprintStrategy;
use cubek_matmul::routines::vec2mat::{Vec2MatRoutine, Vec2MatStrategy};

use crate::suite::layout_to_stride_spec;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_matmul::definition::{MatmulElems, MatmulIdent, MatmulProblem};
use cubek_std::InputBinding;
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

/// Regression test: launch_vec2mat::launch_ref used to hardcode batch dims as
/// shape![1], which caused a sequence length mismatch panic when the actual
/// tensors were 2D (no batch dimension). The BatchLayout's batch_shape had 1
/// element but batch_strides had 0 elements, crashing in the #[unroll] loop.
#[test]
pub fn test_2d_no_batch_via_launch_ref() {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let n = plane_size * 4;
    let k = plane_size * 4;

    // 2D shapes: no batch dimension
    let lhs_shape = shape![1, k];
    let rhs_shape = shape![k, n];
    let out_shape = shape![1, n];

    let global_elems = elems();
    let all_elems = MatmulElems::from_globals(&global_elems);

    let (lhs, lhs_data) = TestInput::new(
        client.clone(),
        lhs_shape.clone(),
        global_elems.lhs,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Random {
            seed: 1234,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (rhs, rhs_data) = TestInput::new(
        client.clone(),
        rhs_shape.clone(),
        global_elems.rhs,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Random {
            seed: 5678,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let out = TestInput::new(
        client.clone(),
        out_shape.clone(),
        global_elems.out,
        layout_to_stride_spec(MatrixLayout::RowMajor),
        DataKind::Zeros,
    )
    .generate_without_host_data();

    let lhs_strides = lhs.strides().clone();
    let rhs_strides = rhs.strides().clone();
    let out_strides = out.strides().clone();

    let lhs_handle = InputBinding::Normal(lhs.binding(), global_elems.lhs);
    let rhs_handle = InputBinding::Normal(rhs.binding(), global_elems.rhs);
    let out_handle = out.clone().binding();

    launch_vec2mat::launch_ref(&client, lhs_handle, rhs_handle, out_handle, &all_elems).unwrap();

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape,
        rhs_shape,
        out_shape,
        lhs_strides,
        rhs_strides,
        out_strides,
        global_elems,
        AddressType::U32,
        None,
        None,
    )
    .unwrap();

    assert_result(&lhs_data, &rhs_data, &problem, &client, out, all_elems);
}

fn test_vec2mat(case: Vec2MatTestCase) {
    let client = TestRuntime::client(&Default::default());
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let problem = case.into_problem(plane_size);

    test_matmul_strategy(
        client,
        problem,
        Strategy::Vec2Mat(BlueprintStrategy::Inferred(Vec2MatStrategy {
            target_num_planes: 8,
        })),
    );
}
