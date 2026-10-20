use cubecl::{
    ir::{ComplexKind, ElemType},
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::{ComplexMatmulOptions, launch_c32_ref},
    strategy::Strategy,
};
use cubek_std::InputBinding;
use num_complex::Complex32;

const C32: ElemType = ElemType::Complex(ComplexKind::C32);

#[test]
fn c32_matmul_is_finite_and_preserves_descriptor_for_reuse() {
    let client = cubecl::test_device().client();
    let lhs_values = [
        Complex32::new(1.0, 2.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(-1.0, 0.5),
        Complex32::new(0.5, 3.0),
        Complex32::new(-2.0, 1.0),
        Complex32::new(1.5, -2.0),
    ];
    let rhs_values = [
        Complex32::new(1.0, -1.0),
        Complex32::new(0.5, 2.0),
        Complex32::new(-2.0, 0.25),
        Complex32::new(3.0, -1.5),
        Complex32::new(0.25, 1.0),
        Complex32::new(-1.0, 0.5),
    ];
    let lhs = tensor(&client, &[2, 3], &[3, 1], &lhs_values);
    let rhs = tensor(&client, &[3, 2], &[2, 1], &rhs_values);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    let original = dtypes.clone();

    for lhs_conj in [false, true] {
        for rhs_conj in [false, true] {
            let out = TensorHandle::empty(&client, [2, 2], C32);
            launch_c32_ref(
                &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
                &client,
                InputBinding::new(lhs.clone().binding(), C32),
                InputBinding::new(rhs.clone().binding(), C32),
                out.clone().binding(),
                &mut dtypes,
                ComplexMatmulOptions { lhs_conj, rhs_conj },
            )
            .unwrap();

            assert_eq!(dtypes, original);
            let bytes = client.read_one_unchecked(out.handle.clone());
            let actual = Complex32::from_bytes(&bytes);
            for row in 0..2 {
                for col in 0..2 {
                    let expected: Complex32 = (0..3)
                        .map(|k| {
                            let lhs = lhs_values[row * 3 + k];
                            let rhs = rhs_values[k * 2 + col];
                            (if lhs_conj {
                                Complex32::new(lhs.re, -lhs.im)
                            } else {
                                lhs
                            }) * (if rhs_conj {
                                Complex32::new(rhs.re, -rhs.im)
                            } else {
                                rhs
                            })
                        })
                        .sum();
                    let value = actual[row * 2 + col];
                    assert!(value.re.is_finite() && value.im.is_finite());
                    assert!(
                        (value.re - expected.re).abs() < 1e-4,
                        "{lhs_conj:?} {rhs_conj:?} {value:?} != {expected:?}"
                    );
                    assert!(
                        (value.im - expected.im).abs() < 1e-4,
                        "{lhs_conj:?} {rhs_conj:?} {value:?} != {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn c32_matmul_respects_column_major_strides() {
    let client = cubecl::test_device().client();
    let lhs_values = [
        Complex32::new(1.0, 1.0),
        Complex32::new(2.0, -3.0),
        Complex32::new(-1.0, 0.5),
        Complex32::new(0.25, 2.0),
    ];
    let rhs_values = [
        Complex32::new(3.0, -1.0),
        Complex32::new(0.5, 1.0),
        Complex32::new(1.0, 2.0),
        Complex32::new(2.0, -1.0),
    ];
    let lhs = tensor(&client, &[2, 2], &[1, 2], &lhs_values);
    let rhs = tensor(&client, &[2, 2], &[1, 2], &rhs_values);
    let out = TensorHandle::new(client.empty(4 * 8), vec![2, 2], vec![1, 2], C32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    launch_c32_ref(
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
        &client,
        InputBinding::new(lhs.binding(), C32),
        InputBinding::new(rhs.binding(), C32),
        out.clone().binding(),
        &mut dtypes,
        ComplexMatmulOptions::default(),
    )
    .unwrap();

    let bytes = client.read_one_unchecked(out.handle);
    let actual = Complex32::from_bytes(&bytes);
    for col in 0..2 {
        for row in 0..2 {
            let expected: Complex32 = (0..2)
                .map(|k| lhs_values[row + k * 2] * rhs_values[k + col * 2])
                .sum();
            let value = actual[row + col * 2];
            assert!((value.re - expected.re).abs() < 1e-4);
            assert!((value.im - expected.im).abs() < 1e-4);
        }
    }
}

#[test]
fn c32_matmul_handles_batched_inputs() {
    let client = cubecl::test_device().client();
    let lhs_values: Vec<_> = (0..8)
        .map(|i| Complex32::new(i as f32 + 0.5, 0.25 - i as f32))
        .collect();
    let rhs_values: Vec<_> = (0..8)
        .map(|i| Complex32::new(1.5 - i as f32 * 0.25, i as f32 + 1.0))
        .collect();
    let lhs = tensor(&client, &[2, 2, 2], &[4, 2, 1], &lhs_values);
    let rhs = tensor(&client, &[2, 2, 2], &[4, 2, 1], &rhs_values);
    let out = TensorHandle::empty(&client, [2, 2, 2], C32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    launch_c32_ref(
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
        &client,
        InputBinding::new(lhs.binding(), C32),
        InputBinding::new(rhs.binding(), C32),
        out.clone().binding(),
        &mut dtypes,
        ComplexMatmulOptions::default(),
    )
    .unwrap();
    let bytes = client.read_one_unchecked(out.handle);
    let actual = Complex32::from_bytes(&bytes);
    for batch in 0..2 {
        for row in 0..2 {
            for col in 0..2 {
                let expected: Complex32 = (0..2)
                    .map(|k| {
                        lhs_values[batch * 4 + row * 2 + k] * rhs_values[batch * 4 + k * 2 + col]
                    })
                    .sum();
                let value = actual[batch * 4 + row * 2 + col];
                assert!((value.re - expected.re).abs() < 1e-4);
                assert!((value.im - expected.im).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn c32_matmul_keeps_noncontiguous_padding() {
    let client = cubecl::test_device().client();
    let padding = Complex32::new(-91.0, 57.0);
    let lhs = tensor(
        &client,
        &[2, 2],
        &[3, 1],
        &[
            Complex32::new(1.0, 2.0),
            Complex32::new(2.0, -1.0),
            padding,
            Complex32::new(3.0, 1.0),
            Complex32::new(-4.0, 0.5),
        ],
    );
    let rhs = tensor(
        &client,
        &[2, 2],
        &[2, 1],
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
    );
    let raw = client.create_from_slice(Complex32::as_bytes(&[padding; 5]));
    let out = TensorHandle::new(raw.clone(), vec![2, 2], vec![3, 1], C32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    launch_c32_ref(
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
        &client,
        InputBinding::new(lhs.binding(), C32),
        InputBinding::new(rhs.binding(), C32),
        out.binding(),
        &mut dtypes,
        ComplexMatmulOptions::default(),
    )
    .unwrap();
    let bytes = client.read_one_unchecked(raw);
    let actual = Complex32::from_bytes(&bytes);
    assert_eq!(actual[2], padding);
    for (index, expected) in [
        (0, Complex32::new(1.0, 2.0)),
        (1, Complex32::new(2.0, -1.0)),
        (3, Complex32::new(3.0, 1.0)),
        (4, Complex32::new(-4.0, 0.5)),
    ] {
        assert!((actual[index].re - expected.re).abs() < 1e-4);
        assert!((actual[index].im - expected.im).abs() < 1e-4);
    }
}

#[test]
fn c32_matmul_broadcasts_unbatched_operand_over_batches() {
    let client = cubecl::test_device().client();
    let identity = [
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    let values: Vec<_> = (0..8)
        .map(|i| Complex32::new(i as f32 - 3.0, i as f32 * 0.25))
        .collect();
    for unbatched_lhs in [true, false] {
        let (lhs, rhs) = if unbatched_lhs {
            (
                tensor(&client, &[2, 2], &[2, 1], &identity),
                tensor(&client, &[2, 2, 2], &[4, 2, 1], &values),
            )
        } else {
            (
                tensor(&client, &[2, 2, 2], &[4, 2, 1], &values),
                tensor(&client, &[2, 2], &[2, 1], &identity),
            )
        };
        let out = TensorHandle::empty(&client, [2, 2, 2], C32);
        let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: C32,
            rhs: C32,
            out: C32,
        });
        launch_c32_ref(
            &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
            &client,
            InputBinding::new(lhs.binding(), C32),
            InputBinding::new(rhs.binding(), C32),
            out.clone().binding(),
            &mut dtypes,
            ComplexMatmulOptions::default(),
        )
        .unwrap();
        let bytes = client.read_one_unchecked(out.handle);
        for (got, expected) in Complex32::from_bytes(&bytes).iter().zip(&values) {
            assert!((got.re - expected.re).abs() < 1e-4);
            assert!((got.im - expected.im).abs() < 1e-4);
        }
    }
}

#[test]
fn c32_matmul_broadcasts_batched_inputs() {
    let client = cubecl::test_device().client();
    let lhs = tensor(
        &client,
        &[1, 2, 2],
        &[4, 2, 1],
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
    );
    let values: Vec<_> = (0..8)
        .map(|i| Complex32::new(i as f32 - 3.0, i as f32 * 0.25))
        .collect();
    let rhs = tensor(&client, &[2, 2, 2], &[4, 2, 1], &values);
    let out = TensorHandle::empty(&client, [2, 2, 2], C32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    launch_c32_ref(
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
        &client,
        InputBinding::new(lhs.binding(), C32),
        InputBinding::new(rhs.binding(), C32),
        out.clone().binding(),
        &mut dtypes,
        ComplexMatmulOptions::default(),
    )
    .unwrap();
    let bytes = client.read_one_unchecked(out.handle);
    for (got, expected) in Complex32::from_bytes(&bytes).iter().zip(values) {
        assert!((got.re - expected.re).abs() < 1e-4);
        assert!((got.im - expected.im).abs() < 1e-4);
    }
}

#[test]
fn c32_matmul_honors_trimmed_bindings() {
    let client = cubecl::test_device().client();
    let identity = [
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    let rhs_values = [
        Complex32::new(1.0, 2.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(-3.0, 0.5),
        Complex32::new(4.0, -2.0),
    ];
    let sentinel = Complex32::new(-20.0, 77.0);
    let lhs_raw = client.create_from_slice(Complex32::as_bytes(&[
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        identity[0],
        identity[1],
        identity[2],
        identity[3],
        sentinel,
    ]));
    let lhs = TensorHandle::new(
        lhs_raw.offset_start(32).offset_end(8),
        vec![2, 2],
        vec![2, 1],
        C32,
    );
    let rhs = tensor(&client, &[2, 2], &[2, 1], &rhs_values);
    let out_raw = client.create_from_slice(Complex32::as_bytes(&[sentinel; 9]));
    let out = TensorHandle::new(
        out_raw.clone().offset_start(32).offset_end(8),
        vec![2, 2],
        vec![2, 1],
        C32,
    );
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    launch_c32_ref(
        &Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive),
        &client,
        InputBinding::new(lhs.binding(), C32),
        InputBinding::new(rhs.binding(), C32),
        out.binding(),
        &mut dtypes,
        ComplexMatmulOptions::default(),
    )
    .unwrap();
    let bytes = client.read_one_unchecked(out_raw);
    let values = Complex32::from_bytes(&bytes);
    assert_eq!(&values[..4], &[sentinel; 4]);
    assert_eq!(values[8], sentinel);
    for (got, expected) in values[4..8].iter().zip(rhs_values) {
        assert!((got.re - expected.re).abs() < 1e-4);
        assert!((got.im - expected.im).abs() < 1e-4);
    }
}

#[test]
fn c32_matmul_rejects_mismatched_shapes_and_storage() {
    let client = cubecl::test_device().client();
    let strategy = Strategy::MultiLevel(cubek_matmul::multi_level::Strategy::Naive);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: C32,
        rhs: C32,
        out: C32,
    });
    for (lhs, rhs, out) in [
        (vec![2, 3], vec![2, 2], vec![2, 2]),          // K mismatch
        (vec![2, 3], vec![3, 2], vec![3, 2]),          // M mismatch
        (vec![2, 2, 2], vec![3, 2, 2], vec![3, 2, 2]), // incompatible batches
    ] {
        assert!(
            launch_c32_ref(
                &strategy,
                &client,
                InputBinding::new(TensorHandle::empty(&client, lhs, C32).binding(), C32),
                InputBinding::new(TensorHandle::empty(&client, rhs, C32).binding(), C32),
                TensorHandle::empty(&client, out, C32).binding(),
                &mut dtypes,
                ComplexMatmulOptions::default(),
            )
            .is_err()
        );
    }
    let lhs = TensorHandle::new(client.empty(3 * 8), vec![2, 2], vec![2, 1], C32);
    let rhs = TensorHandle::empty(&client, [2, 2], C32);
    let out = TensorHandle::empty(&client, [2, 2], C32);
    assert!(
        launch_c32_ref(
            &strategy,
            &client,
            InputBinding::new(lhs.binding(), C32),
            InputBinding::new(rhs.binding(), C32),
            out.binding(),
            &mut dtypes,
            ComplexMatmulOptions::default(),
        )
        .is_err()
    );
}

fn tensor(
    client: &Client,
    shape: &[usize],
    strides: &[usize],
    values: &[Complex32],
) -> TensorHandle {
    TensorHandle::new(
        client.create_from_slice(Complex32::as_bytes(values)),
        shape.to_vec(),
        strides.to_vec(),
        C32,
    )
}
