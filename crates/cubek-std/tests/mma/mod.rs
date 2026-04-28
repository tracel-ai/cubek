use cubecl::{
    features::MmaConfig,
    ir::MatrixIdent,
    prelude::*,
    {self, Runtime, TestRuntime},
};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, ValidationResult, pretty_print_zip,
};

#[cube(launch)]
// F should be either AB or CD to match the ident
pub fn mma_layout_kernel<AB: Numeric, CD: Numeric, F: Numeric>(
    lane_tensor: &mut Tensor<F>,
    vector_tensor: &mut Tensor<F>,
    within_vector_tensor: &mut Tensor<F>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] stride: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, CD>::new(m, n, k);
    let lane_id = UNIT_POS_PLANE as usize;

    let vector_count = def.vectors_per_lane(MatrixIdent::A);
    let vector_size = def.vector_size(MatrixIdent::A);

    #[unroll]
    for i in 0..vector_count {
        #[unroll]
        for j in 0..vector_size {
            let nth = i * vector_size + j;
            let (row, col) =
                def.position_of_nth(lane_id as u32, nth as u32, MatrixIdent::Accumulator);

            let absolute_index = row as usize * stride + col as usize;

            lane_tensor[absolute_index] = F::cast_from(lane_id);
            vector_tensor[absolute_index] = F::cast_from(i);
            within_vector_tensor[absolute_index] = F::cast_from(j);
        }
    }
}

pub fn print_mma_layout<AB: CubeElement + Numeric, CD: CubeElement + Numeric>(
    m: usize,
    n: usize,
    k: usize,
    ident: MatrixIdent,
) -> TestOutcome {
    let client = TestRuntime::client(&Default::default());

    if !client
        .properties()
        .features
        .matmul
        .mma
        .contains(&MmaConfig {
            a_type: AB::cube_type(),
            b_type: AB::cube_type(),
            cd_type: CD::cube_type(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
        })
    {
        return TestOutcome::CompileError(format!(
            "MmaConfig not available for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            AB::cube_type(),
            AB::cube_type(),
            CD::cube_type()
        ));
    }

    let (rows, cols, dtype) = match ident {
        MatrixIdent::A => (m, k, AB::as_type_native_unchecked().storage_type()),
        MatrixIdent::B => (k, n, AB::as_type_native_unchecked().storage_type()),
        MatrixIdent::Accumulator => (m, n, CD::as_type_native_unchecked().storage_type()),
    };

    let lane_tensor = TestInput::builder(client.clone(), [rows, cols])
        .dtype(dtype)
        .zeros()
        .generate();

    let vector_tensor = TestInput::builder(client.clone(), [rows, cols])
        .dtype(dtype)
        .zeros()
        .generate();

    let within_vector_tensor = TestInput::builder(client.clone(), [rows, cols])
        .dtype(dtype)
        .zeros()
        .generate();

    match ident {
        MatrixIdent::A | MatrixIdent::B => {
            mma_layout_kernel::launch::<AB, CD, AB, TestRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(client.properties().hardware.plane_size_max),
                lane_tensor.clone().binding().into_tensor_arg(),
                vector_tensor.clone().binding().into_tensor_arg(),
                within_vector_tensor.clone().binding().into_tensor_arg(),
                m,
                n,
                k,
                cols,
            );
        }
        MatrixIdent::Accumulator => {
            mma_layout_kernel::launch::<AB, CD, CD, TestRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(client.properties().hardware.plane_size_max),
                lane_tensor.clone().binding().into_tensor_arg(),
                vector_tensor.clone().binding().into_tensor_arg(),
                within_vector_tensor.clone().binding().into_tensor_arg(),
                m,
                n,
                k,
                cols,
            );
        }
    }

    let lane_tensor = HostData::from_tensor_handle(&client, lane_tensor, HostDataType::I32);
    let vector_tensor = HostData::from_tensor_handle(&client, vector_tensor, HostDataType::I32);
    let within_vector_tensor =
        HostData::from_tensor_handle(&client, within_vector_tensor, HostDataType::I32);

    let table = pretty_print_zip(&[&lane_tensor, &vector_tensor, &within_vector_tensor]);

    TestOutcome::Validated(ValidationResult::Fail(format!("Mma Layout:\n{}", table)))
}

use half::{bf16, f16};

#[test]
fn print_a_f16_f32_m16n8k16() {
    print_mma_layout::<f16, f32>(16, 8, 16, MatrixIdent::A).enforce();
}

#[test]
fn print_a_bf16_f32_m16n8k16() {
    print_mma_layout::<bf16, f32>(16, 8, 16, MatrixIdent::A).enforce();
}

#[test]
fn print_a_f16_f32_m16n8k8() {
    print_mma_layout::<f16, f32>(16, 8, 8, MatrixIdent::A).enforce();
}

#[test]
fn print_b_f16_f32_m16n8k16() {
    print_mma_layout::<f16, f32>(16, 8, 16, MatrixIdent::B).enforce();
}

#[test]
fn print_b_bf16_f32_m16n8k16() {
    print_mma_layout::<bf16, f32>(16, 8, 16, MatrixIdent::B).enforce();
}

#[test]
fn print_b_f16_f32_m16n8k8() {
    print_mma_layout::<f16, f32>(16, 8, 8, MatrixIdent::B).enforce();
}

#[test]
fn print_acc_f16_f32_m16n8k16() {
    print_mma_layout::<f16, f32>(16, 8, 16, MatrixIdent::Accumulator).enforce();
}

#[test]
fn print_acc_bf16_f32_m16n8k16() {
    print_mma_layout::<bf16, f32>(16, 8, 16, MatrixIdent::Accumulator).enforce();
}

#[test]
fn print_acc_f16_f32_m16n8k8() {
    print_mma_layout::<f16, f32>(16, 8, 8, MatrixIdent::Accumulator).enforce();
}
