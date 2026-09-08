//! Packing a matrix into storage tiles and back: the packed buffer holds the tiles it states,
//! the round trip is the identity, and what cannot be packed is refused on the host.

use cubecl::{ir::ElemType, prelude::*, zspace::Shape};
use cubek_matmul::{
    definition::MatmulSetupError,
    tiled::pack::{pack, unpack},
};
use cubek_test_utils::{HostData, HostDataType, TestInput, client};

/// Pack, check every cell sits in its tile, unpack, check every cell is back.
fn round_trip(batches: &[usize], rows: usize, cols: usize, (tr, tc): (usize, usize)) {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let shape: Vec<usize> = batches.iter().copied().chain([rows, cols]).collect();
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(shape.clone()))
        .dtype(dtype)
        .custom(data.clone())
        .generate_with_f32_host_data();

    let packed = pack(&client, src.binding(), dtype, (tr, tc)).unwrap();
    let physical: Vec<usize> = batches
        .iter()
        .copied()
        .chain([rows / tr, cols / tc, tr, tc])
        .collect();
    assert_eq!(packed.shape().as_slice(), &physical[..]);
    assert!(packed.metadata.is_tiled());
    assert_eq!(
        packed.metadata.logical_shape().unwrap().as_slice(),
        &shape[..]
    );

    let got = HostData::from_tensor_handle(&client, packed.clone(), HostDataType::F32);
    let nb: usize = batches.iter().product();
    for b in 0..nb {
        // The batch coordinate, outermost first.
        let mut coord = Vec::with_capacity(batches.len() + 4);
        let mut rest = b;
        for &extent in batches.iter().rev() {
            coord.push(rest % extent);
            rest /= extent;
        }
        coord.reverse();
        for r in 0..rows {
            for c in 0..cols {
                let want = data[(b * rows + r) * cols + c];
                let mut at = coord.clone();
                at.extend([r / tr, c / tc, r % tr, c % tc]);
                let have = got.get_f32(&at);
                assert_eq!(have, want, "batch {b} cell ({r}, {c}) packed at {at:?}");
            }
        }
    }

    let back = unpack(&client, packed.binding(), dtype).unwrap();
    assert_eq!(back.shape().as_slice(), &shape[..]);
    assert!(!back.metadata.is_tiled());
    let got = HostData::from_tensor_handle(&client, back, HostDataType::F32);
    for b in 0..nb {
        let mut coord = Vec::with_capacity(batches.len() + 2);
        let mut rest = b;
        for &extent in batches.iter().rev() {
            coord.push(rest % extent);
            rest /= extent;
        }
        coord.reverse();
        for r in 0..rows {
            for c in 0..cols {
                let want = data[(b * rows + r) * cols + c];
                let mut at = coord.clone();
                at.extend([r, c]);
                assert_eq!(got.get_f32(&at), want, "batch {b} cell ({r}, {c}) unpacked");
            }
        }
    }
}

#[test]
fn pack_stores_one_tile_at_a_time() {
    round_trip(&[], 64, 96, (16, 32));
}

#[test]
fn pack_tiles_a_narrow_column_run() {
    round_trip(&[], 128, 64, (32, 8));
}

#[test]
fn pack_keeps_batches_plain() {
    round_trip(&[3], 32, 64, (16, 16));
}

#[test]
fn pack_keeps_two_batch_dims_plain() {
    round_trip(&[2, 3], 32, 32, (8, 16));
}

#[test]
fn pack_refuses_a_tile_that_does_not_divide() {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![48, 64]))
        .dtype(dtype)
        .custom((0..48 * 64).map(|i| i as f32).collect())
        .generate_with_f32_host_data();
    assert!(matches!(
        pack(&client, src.binding(), dtype, (32, 32)),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
}

#[test]
fn pack_refuses_a_packed_source_and_unpack_a_plain_one() {
    let client = client();
    let dtype: ElemType = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), Shape::from(vec![32, 32]))
        .dtype(dtype)
        .custom((0..32 * 32).map(|i| i as f32).collect())
        .generate_with_f32_host_data();
    assert!(matches!(
        unpack(&client, src.clone().binding(), dtype),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
    let packed = pack(&client, src.binding(), dtype, (16, 16)).unwrap();
    assert!(matches!(
        pack(&client, packed.binding(), dtype, (16, 16)),
        Err(MatmulSetupError::InvalidConfig(_))
    ));
}
