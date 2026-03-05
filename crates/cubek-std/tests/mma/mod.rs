use cubecl::features::MmaConfig;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{self, Runtime, TestRuntime};
use cubek_test_utils::{
    DataKind, HostData, HostDataType, StrideSpec, TestInput, TestOutcome, ValidationResult,
};

#[cube(launch)]
pub fn reverse_engineer_layout_kernel<AB: Numeric, CD: Numeric>(
    lane_tensor: &mut Tensor<AB>,
    nth_tensor: &mut Tensor<AB>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] stride: usize,
) {
    let def = cmma::MmaDefinition::<AB, AB, CD>::new(m, n, k);
    let lane_id = UNIT_POS_PLANE as usize;

    let line_count = def.lines_per_lane(MatrixIdent::A);
    let line_size = def.line_size(MatrixIdent::A);

    #[unroll]
    for i in 0..line_count {
        #[unroll]
        for j in 0..line_size {
            let nth = i * line_size + j;
            let (row, col) =
                def.position_of_nth(lane_id as u32, nth as u32, MatrixIdent::Accumulator);
            let absolute_index = row as usize * stride + col as usize;
            lane_tensor[absolute_index] = AB::cast_from(lane_id);
            nth_tensor[absolute_index] = AB::cast_from(nth);
        }
    }
}

pub fn reverse_engineer_layout<AB: CubeElement + Numeric, CD: CubeElement + Numeric>(
    m: usize,
    n: usize,
    k: usize,
    ident: MatrixIdent,
) -> TestOutcome {
    let client = TestRuntime::client(&Default::default());

    if !client.properties().features.mma.contains(&MmaConfig {
        a_type: AB::cube_type(),
        b_type: AB::cube_type(),
        cd_type: CD::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    }) {
        return TestOutcome::CompileError(format!(
            "MmaConfig not available for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            AB::cube_type(),
            AB::cube_type(),
            CD::cube_type()
        ));
    }

    let (rows, cols, dtype) = match ident {
        MatrixIdent::A => (m, k, AB::as_type_native_unchecked()),
        MatrixIdent::B => (k, n, AB::as_type_native_unchecked()),
        MatrixIdent::Accumulator => (m, n, CD::as_type_native_unchecked()),
    };

    let lane_tensor = TestInput::new(
        client.clone(),
        [rows, cols],
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate();

    let nth_tensor = TestInput::new(
        client.clone(),
        [rows, cols],
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate();

    reverse_engineer_layout_kernel::launch::<AB, CD, TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(client.properties().hardware.plane_size_max),
        lane_tensor.as_ref().as_tensor_arg(1),
        nth_tensor.as_ref().as_tensor_arg(1),
        m,
        n,
        k,
        cols,
    )
    .unwrap();

    let lane_tensor = HostData::from_tensor_handle(&client, &lane_tensor, HostDataType::F32);
    let nth_tensor = HostData::from_tensor_handle(&client, &nth_tensor, HostDataType::F32);

    TestOutcome::Validated(ValidationResult::Fail(format!(
        "lane_tensor: {:?}, nth_tensor {:?}",
        lane_tensor, nth_tensor,
    )))
}

use half::{bf16, f16};

#[test]
fn rev_eng_a_f16_f32_m16n8k16() {
    reverse_engineer_layout::<f16, f32>(16, 8, 16, MatrixIdent::A).enforce();
}

#[test]
fn rev_eng_a_bf16_f32_m16n8k16() {
    reverse_engineer_layout::<bf16, f32>(16, 8, 16, MatrixIdent::A).enforce();
}

#[test]
fn rev_eng_a_f16_f32_m16n8k8() {
    reverse_engineer_layout::<f16, f32>(16, 8, 8, MatrixIdent::A).enforce();
}

#[test]
fn rev_eng_b_f16_f32_m16n8k16() {
    reverse_engineer_layout::<f16, f32>(16, 8, 16, MatrixIdent::B).enforce();
}

#[test]
fn rev_eng_b_bf16_f32_m16n8k16() {
    reverse_engineer_layout::<bf16, f32>(16, 8, 16, MatrixIdent::B).enforce();
}

#[test]
fn rev_eng_b_f16_f32_m16n8k8() {
    reverse_engineer_layout::<f16, f32>(16, 8, 8, MatrixIdent::B).enforce();
}

#[test]
fn rev_eng_acc_f16_f32_m16n8k16() {
    reverse_engineer_layout::<f16, f32>(16, 8, 16, MatrixIdent::Accumulator).enforce();
}

#[test]
fn rev_eng_acc_bf16_f32_m16n8k16() {
    reverse_engineer_layout::<bf16, f32>(16, 8, 16, MatrixIdent::Accumulator).enforce();
}

#[test]
fn rev_eng_acc_f16_f32_m16n8k8() {
    reverse_engineer_layout::<f16, f32>(16, 8, 8, MatrixIdent::Accumulator).enforce();
}
