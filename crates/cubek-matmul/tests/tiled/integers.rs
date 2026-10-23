//! An `i32` product sums in `i32`: every setting of a routine that serves it writes the exact
//! product, and one that cannot serve it declines.

use cubecl::ir::{ElemType, IntKind};
use cubecl::zspace::shape;
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, MatmulRoutine, MatmulRoutineCandidate,
    PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine, UnitWholeKMatmulRoutine,
    tensor::PlainTensor,
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_test_utils::TestInput;

const I32: ElemType = ElemType::Int(IntKind::I32);

/// Small integers, distinct enough that a wrong pairing of rows and columns changes the sum.
fn values(len: usize, seed: i32) -> Vec<i32> {
    (0..len as i32).map(|i| (i * 7 + seed) % 11 - 5).collect()
}

/// `lhs · rhs` on the host, both row-major.
fn product(lhs: &[i32], rhs: &[i32], [m, n, k]: [usize; 3]) -> Vec<i32> {
    (0..m * n)
        .map(|at| {
            let (row, column) = (at / n, at % n);
            (0..k).map(|i| lhs[row * k + i] * rhs[i * n + column]).sum()
        })
        .collect()
}

/// Every setting `routine` lists, launched on an `m × k` by `k × n` product of `i32`: the ones
/// that serve it write the exact product. How many served, so a caller can say a routine must.
fn every_setting_is_exact<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    routine: R,
    [m, n, k]: [usize; 3],
) -> usize
where
    R: MatmulRoutine<M, SM> + Copy,
{
    let client = cubecl::test_device().client();
    let hw = Hardware::new(client.properties_shared(), client.target_properties());
    let (lhs_host, rhs_host) = (values(m * k, 3), values(k * n, 5));
    let expected = product(&lhs_host, &rhs_host, [m, n, k]);
    let upload = |shape, data: &[i32]| {
        let data = data.iter().map(|&value| value as f32).collect();
        TestInput::builder(client.clone(), shape)
            .dtype(I32)
            .custom(data)
            .generate_without_host_data()
    };
    let lhs = upload(shape![m, k], &lhs_host);
    let rhs = upload(shape![k, n], &rhs_host);
    let mut served = 0;
    for candidate in MatmulRoutineCandidate::list(routine, &hw) {
        let (out, out_plain) = PlainTensor::empty(&client, shape![m, n], I32);
        let launched = routine.launch_into(
            &hw,
            &client,
            PlainTensor::of_handle(lhs.clone()),
            PlainTensor::of_handle(rhs.clone()),
            out_plain,
            candidate.settings(),
        );
        if launched.is_err() {
            continue;
        }
        served += 1;
        let written: Vec<i32> = client
            .read_one_unchecked(out.handle)
            .chunks_exact(4)
            .take(m * n)
            .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        assert_eq!(
            written,
            expected,
            "{} writes a wrong i32 product",
            candidate.settings()
        );
    }
    served
}

#[test]
fn unit_registers_sums_i32_exactly() {
    assert!(every_setting_is_exact(UnitRegistersMatmulRoutine, [32, 24, 40]) > 0);
}

#[test]
fn unit_whole_k_sums_i32_exactly() {
    every_setting_is_exact(UnitWholeKMatmulRoutine, [4, 24, 40]);
}

#[test]
fn plane_shuffle_sums_i32_exactly() {
    every_setting_is_exact(PlaneShuffleMatmulRoutine, [4, 24, 40]);
}

#[test]
fn cube_shuffle_sums_i32_exactly() {
    every_setting_is_exact(CubeShuffleMatmulRoutine, [4, 24, 40]);
}

#[test]
fn cube_staged_sums_i32_exactly() {
    every_setting_is_exact(CubeStagedMatmulRoutine, [32, 24, 40]);
}
