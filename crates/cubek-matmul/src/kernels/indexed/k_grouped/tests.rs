//! The K-grouped launch against a host reference.

use super::{KGroupedMatmulTensors, launch};
use crate::kernels::indexed::m_grouped::upload;
use crate::kernels::tensor::PlainTensor;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubek_std::{Hardware, Instructions};
use cubek_test_utils::client;

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// `values` uploaded as a row-major `f32` tensor of `shape`.
fn f32s(client: &Client, shape: [usize; 2], values: &[f32]) -> PlainTensor {
    let handle = client.create_from_slice(f32::as_bytes(values));
    PlainTensor::of_handle(TensorHandle::new_contiguous(shape, handle, F32))
}

/// The K-grouped product of `lhs` and `rhs`, their rows grouped by `ends`.
fn launch_grouped(
    client: &Client,
    lhs: PlainTensor,
    rhs: PlainTensor,
    ends: &[u32],
) -> Result<TensorHandle, Decline> {
    let ends = PlainTensor::of_handle(upload(client, ends));
    launch(&KGroupedMatmulTensors::new(client, lhs, rhs, ends)?)
}

/// Each group's rows of two operands contracted against each other into
/// its slab of a stack, against a host reference: ragged groups, an empty
/// one whose slab is zero, an empty first group, one spanning several
/// stages, groups whose rows start and end partway into a stage, a single
/// group, and extents no vector divides.
#[test]
fn each_group_contracts_its_own_rows_into_its_slab() {
    let client = client();
    let offers_f32 = Hardware::from_client(&client)
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .for_triple(F32, F32, F32)
        .next()
        .is_some();
    let cases: [(usize, usize, &[usize]); 5] = [
        (64, 64, &[5, 0, 37, 2]),
        (128, 96, &[5, 0, 37, 2]),
        (13, 11, &[5, 0, 11, 2]),
        (64, 48, &[0, 7, 3]),
        (32, 64, &[40]),
    ];
    for (k, n, sizes) in cases {
        let groups = sizes.len();
        let rows: usize = sizes.iter().sum();
        let ends: Vec<u32> = sizes
            .iter()
            .scan(0, |end, size| {
                *end += size;
                Some(*end as u32)
            })
            .collect();
        let group_of = |r: usize| ends.iter().position(|&end| r < end as usize).unwrap();
        let x: Vec<f32> = (0..rows * k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let g: Vec<f32> = (0..rows * n).map(|i| ((i % 5) as f32) - 2.0).collect();

        // A device with no f32 instruction declines, and the caller's
        // dense form answers there; one that offers it runs every case.
        let out = match launch_grouped(
            &client,
            f32s(&client, [rows, k], &x),
            f32s(&client, [rows, n], &g),
            &ends,
        ) {
            Ok(out) => out,
            Err(why) => {
                assert!(
                    !offers_f32,
                    "k {k} n {n} groups {sizes:?} declined on a device with an f32 \
                     instruction: {why:?}"
                );
                continue;
            }
        };
        assert_eq!(out.shape().to_vec(), vec![k, groups * n]);
        let bytes = client.read_one(out.handle).expect("the output reads back");
        let got = f32::from_bytes(&bytes);
        for e in 0..groups {
            for i in 0..k {
                for j in 0..n {
                    let want: f32 = (0..rows)
                        .filter(|&r| group_of(r) == e)
                        .map(|r| x[r * k + i] * g[r * n + j])
                        .sum();
                    assert_eq!(
                        got[i * groups * n + e * n + j],
                        want,
                        "k {k} n {n} groups {sizes:?}: group {e} row {i} column {j}"
                    );
                }
            }
        }
    }
}
