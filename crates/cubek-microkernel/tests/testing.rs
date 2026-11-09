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
    assert!(
        ran > 0,
        "the test device runs no plane fragment microkernel"
    );
}

#[cfg(feature = "attention")]
mod attention {
    use super::{F16, F32, values};
    use cubecl::ir::{ElemType, UIntKind};
    use cubecl::zspace::shape;
    use cubek_microkernel::attention::plane::PlaneFragmentAttentionMicrokernel;
    use cubek_microkernel::testing::attention::{AttentionElements, PlaneFragmentAttentionLaunch};
    use cubek_std::Hardware;
    use cubek_std::microkernel::Microkernel;
    use cubek_test_utils::{HostData, HostDataType, TestInput};

    /// `softmax(scale · q · kᵀ) · v` on the CPU, every key attended, all row-major, and each
    /// query's `ln Σ exp(scale · q · kᵀ)`.
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        [queries, keys]: [usize; 2],
        [head, val]: [usize; 2],
        scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0; queries * val];
        let mut logsumexp = vec![0.0; queries];
        for i in 0..queries {
            let scores: Vec<f32> = (0..keys)
                .map(|j| {
                    scale
                        * (0..head)
                            .map(|d| q[i * head + d] * k[j * head + d])
                            .sum::<f32>()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            logsumexp[i] = max + total.ln();
            for d in 0..val {
                out[i * val + d] =
                    (0..keys).map(|j| weights[j] * v[j * val + d]).sum::<f32>() / total;
            }
        }
        (out, logsumexp)
    }

    fn check(launch: &PlaneFragmentAttentionLaunch) {
        let client = cubecl::test_device().client();
        let [queries, keys] = launch.seq;
        let (head, val) = (launch.item.head_dim, launch.item.val_dim);
        let input = launch.item.elements.query;
        // Small values, so the scores stay in a range `f16` holds and the softmax spreads.
        let scaled = |len, modulo, offset| -> Vec<f32> {
            values(len, modulo, offset)
                .into_iter()
                .map(|x| x / 4.0)
                .collect()
        };
        let (q, k, v) = (
            scaled(queries * head, 7, 3.0),
            scaled(keys * head, 5, 2.0),
            scaled(keys * val, 11, 5.0),
        );
        // The query and output over `[B, KV, G, Q, _]`, the keys and values over `[B, KV, S, _]`,
        // each of one batch and one head.
        let per_head = |data: Vec<f32>, rows, columns, element| {
            TestInput::builder(client.clone(), shape![1, 1, 1, rows, columns])
                .dtype(element)
                .custom(data)
                .generate_without_host_data()
        };
        let per_key_head = |data: Vec<f32>, rows, columns, element| {
            TestInput::builder(client.clone(), shape![1, 1, rows, columns])
                .dtype(element)
                .custom(data)
                .generate_without_host_data()
        };
        let out = per_head(
            vec![-1e4; queries * val],
            queries,
            val,
            launch.item.elements.out,
        );
        let logsumexp = TestInput::builder(client.clone(), shape![1, 1, 1, queries])
            .dtype(launch.item.sum)
            .custom(vec![-1e4; queries])
            .generate_without_host_data();
        let lengths = TestInput::builder(client.clone(), shape![1])
            .dtype(ElemType::UInt(UIntKind::U32))
            .custom(vec![keys as f32])
            .generate_without_host_data();
        let scale = 1.0 / (head as f32).sqrt();
        launch
            .launch(
                &client,
                per_head(q.clone(), queries, head, input).binding(),
                per_key_head(k.clone(), keys, head, input).binding(),
                per_key_head(v.clone(), keys, val, input).binding(),
                out.clone().binding(),
                Some(logsumexp.clone().binding()),
                lengths.binding().into_tensor_arg(),
                scale,
            )
            .unwrap_or_else(|why| panic!("{:?}: {why:?}", launch.microkernel));

        let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        let got_logsumexp = HostData::from_tensor_handle(&client, logsumexp, HostDataType::F32);
        let (want, want_logsumexp) = reference(&q, &k, &v, launch.seq, [head, val], scale);
        for i in 0..queries {
            for d in 0..val {
                let (have, want) = (got.get_f32(&[0, 0, 0, i, d]), want[i * val + d]);
                assert!(
                    (have - want).abs() <= 2e-2 * want.abs().max(1.0),
                    "{:?} at ({i}, {d}): got {have}, want {want}",
                    launch.microkernel
                );
            }
            let (have, want) = (got_logsumexp.get_f32(&[0, 0, 0, i]), want_logsumexp[i]);
            assert!(
                (have - want).abs() <= 2e-2 * want.abs().max(1.0),
                "{:?} logsumexp at {i}: got {have}, want {want}",
                launch.microkernel
            );
        }
    }

    /// Every plane fragment attention microkernel the test device runs, at each of its settings,
    /// over one and three stages of keys, for `f16` and `f32` inputs summed in `f32`.
    #[test]
    fn plane_fragment_attention_matches_the_cpu() {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        let mut ran = 0;
        for input in [F16, F32] {
            let elements = AttentionElements {
                input,
                sum: F32,
                out: F32,
            };
            for settings in PlaneFragmentAttentionMicrokernel::settings(&hw) {
                for steps in [1, 3] {
                    let Ok(launch) = PlaneFragmentAttentionLaunch::new(
                        &hw,
                        elements,
                        [64, 64],
                        &settings,
                        steps,
                    ) else {
                        continue;
                    };
                    check(&launch);
                    ran += 1;
                }
            }
        }
        assert!(
            ran > 0,
            "the test device runs no plane fragment attention microkernel"
        );
    }
}
