//! Microkernels launched on their own on the test device, each against a CPU reference.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::zspace::shape;
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernel;
use cubek_microkernel::testing::matmul::{MatmulElements, PlaneFragmentLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// Small whole numbers, exact in every float element a product here reads and sums in.
fn values(len: usize, modulo: usize, offset: f32) -> Vec<f32> {
    (0..len).map(|i| (i % modulo) as f32 - offset).collect()
}

/// `out = lhs · rhs` on the CPU, both row-major.
fn reference(lhs: &[f32], rhs: &[f32], [m, n, k]: [usize; 3]) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| lhs[i * k + p] * rhs[p * n + j]).sum()
        })
        .collect()
}

/// Runs `launch` on the test device and checks every cell against the CPU.
fn check_plane_fragment(launch: &PlaneFragmentLaunch) {
    let client = cubecl::test_device().client();
    let [m, n, k] = launch.shape;
    let elements = launch.elements;
    let lhs = values(m * k, 7, 3.0);
    let rhs = values(k * n, 5, 2.0);
    let input = |values: Vec<f32>, rows, columns| {
        TestInput::builder(client.clone(), shape![rows, columns])
            .dtype(elements.input)
            .custom(values)
            .generate_without_host_data()
    };
    let (lhs_handle, rhs_handle) = (input(lhs.clone(), m, k), input(rhs.clone(), k, n));
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(elements.out)
        .custom(vec![-1e4; m * n])
        .generate_without_host_data();

    launch
        .launch(
            &client,
            lhs_handle.binding().into_tensor_arg(),
            rhs_handle.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
        )
        .unwrap_or_else(|why| panic!("{:?}: {why:?}", launch.microkernel));

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(&lhs, &rhs, launch.shape);
    for i in 0..m {
        for j in 0..n {
            let (have, want) = (got.get_f32(&[i, j]), want[i * n + j]);
            assert!(
                (have - want).abs() <= 1e-2 * want.abs().max(1.0),
                "{:?} at ({i}, {j}): got {have}, want {want}",
                launch.microkernel
            );
        }
    }
}

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// Every plane fragment microkernel the test device runs, at each of its settings, over one and
/// three stages of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_fragment_matmul_matches_the_cpu() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let mut ran = 0;
    for input in [F16, F32] {
        let elements = MatmulElements {
            input,
            sum: F32,
            out: F32,
        };
        for settings in PlaneFragmentMicrokernel::settings(&hw) {
            for steps in [1, 3] {
                let Ok(launch) = PlaneFragmentLaunch::new(&hw, elements, &settings, steps) else {
                    continue;
                };
                check_plane_fragment(&launch);
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "the test device runs no plane fragment microkernel");
}
