use crate::suite::assert_result;
use crate::suite::layered::matmul_test_launcher::launch_matmul_algorithm;
use cubecl::Runtime;
use cubecl::TestRuntime;
use cubecl::frontend::CubePrimitive;
use cubek_matmul::MatmulInputHandleRef;
use cubek_matmul::components::MatmulElems;
use cubek_matmul::components::MatmulIdent;
use cubek_matmul::components::MatmulProblem;
use cubek_matmul::components::MatmulSelection;
use cubek_matmul::components::MatrixLayout;
use cubek_matmul::components::SwizzleConfig;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::components::{PartitionSize, StageSize, TileSize, TilingScheme};
use cubek_matmul::kernels::layered::simple::SimpleAlgorithm;
use cubek_matmul::kernels::layered::simple_unit::SimpleUnitAlgorithm;
use cubek_matmul::tune_key::MatmulElemType;
use cubek_std::test_utils::TestInput;
use cubek_std::test_utils::batched_matrix_strides;
use cubek_std::test_utils::current_test_mode;

use crate::suite::layered::matmul_test_launcher::InputRepresentation;
use crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;

pub mod matmul_test_launcher;

mod suite;

#[test]
fn small_test_matmul() {
    let client = TestRuntime::client(&Default::default());

    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(TileSize::new(3, 3, 3))
        .with_partition_size(PartitionSize::new(1, 1, 1))
        .with_stage_size(StageSize::new(32, 1, 1))
        .build()
        .unwrap();
    let plane_dim = client.properties().hardware.plane_size_max;
    let selection_builder = MatmulSelection::builder(tiling_scheme, plane_dim);
    let matmul_selection = selection_builder
        .partition_buffering(PartitionBuffering::Single)
        .build();

    let mut problem = MatmulProblem {
        m: 3,
        n: 3,
        k: 3,
        lhs_batches: vec![],
        rhs_batches: vec![],
        out_batches: vec![],
        lhs_strides: vec![],
        rhs_strides: vec![],
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
    };
    let selection = matmul_selection;
    let dtypes = MatmulElems::from_single_dtype(MatmulElemType {
        dtype: f32::as_type_native_unchecked(),
        quantized: false,
    });
    let input_representation = InputRepresentation::Normal;

    let lhs_shape = problem.shape(MatmulIdent::Lhs);
    let rhs_shape = problem.shape(MatmulIdent::Rhs);

    let (lhs, lhs_data) = TestInput::eye(client.clone(), lhs_shape.clone(), *dtypes.lhs_global)
        .generate_with_f32_host_data()
        .unwrap();

    let (rhs, rhs_data) = TestInput::arange(
        client.clone(),
        rhs_shape.clone(),
        *dtypes.rhs_global,
        Some(batched_matrix_strides(
            &rhs_shape,
            matches!(problem.rhs_layout, MatrixLayout::ColMajor),
        )),
    )
    .generate_with_f32_host_data()
    .unwrap();

    let out = TestInput::zeros(
        client.clone(),
        problem.shape(MatmulIdent::Out),
        *dtypes.acc_global,
    )
    .generate_without_host_data()
    .unwrap();

    problem.lhs_strides = lhs.strides.clone();
    problem.rhs_strides = rhs.strides.clone();

    let lhs_handle = MatmulInputHandleRef::Normal(lhs.as_ref(), *dtypes.lhs_global);
    let rhs_handle = MatmulInputHandleRef::Normal(rhs.as_ref(), *dtypes.rhs_global);
    let out_handle = out.as_ref();

    use cubek_matmul::components::tile::io::Filled;
    pub type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

    if launch_matmul_algorithm::<SimpleUnitAlgorithm>(
        &client,
        &problem,
        selection,
        &dtypes,
        input_representation,
        lhs_handle,
        rhs_handle,
        out_handle,
    ) {
        assert_result(&lhs_data, &rhs_data, &problem, &client, &out, dtypes);
    } else {
        if current_test_mode().should_fail_on_test_compilation_fail() {
            panic!("Test did not run ")
        }
    }
}
