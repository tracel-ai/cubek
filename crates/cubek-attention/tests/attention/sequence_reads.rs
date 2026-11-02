//! A decode step of the cube shuffle routine reading each sequence's own length through a page
//! table, against the CPU reference.
//!
//! Pages are scattered across a pool and slots no sequence owns hold NaN, so a read through a
//! wrong slot shows. Pages of 16 and 12 keys reach the step walk; a page that is a larger multiple
//! of a step is walked a page at a time instead.

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::zspace::Shape;
use cubek_attention::kernels::{
    ATTENTION_ELEMENT_UNVERIFIED, AttentionRoutine, AttentionScoring, AttentionTensors,
    AttentionVisibility, CubeShuffleAttentionRoutine, ScoreScale,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, TestInput};

const BATCH: usize = 4;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 64;

fn upload(client: &Client, shape: &[usize], values: Vec<f32>, dtype: ElemType) -> PlainTensor {
    let handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
        .dtype(dtype)
        .custom(values)
        .generate_without_host_data();
    PlainTensor::of_handle(handle)
}

fn run(lengths: [usize; BATCH], page: usize) {
    let client: Client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let (f32_dtype, u32_dtype) = (f32::elem_type_native(), u32::elem_type_native());
    let pages = lengths.map(|length| length.div_ceil(page).max(1));
    let width_pages = pages.into_iter().max().unwrap();
    let width = width_pages * page;
    let draw = |shape: Vec<usize>, seed: u64| {
        TestInput::builder(client.clone(), Shape::from(shape))
            .dtype(f32_dtype)
            .uniform(seed, -1., 1.)
            .generate_with_f32_host_data()
    };
    let (query_handle, query) = draw(vec![BATCH, HEADS, 1, DIM], 1);
    let (_, key) = draw(vec![BATCH, KV_HEADS, width, DIM], 2);
    let (_, value) = draw(vec![BATCH, KV_HEADS, width, DIM], 3);

    // Pages go to scattered slots; the spare ones hold NaN and fill the tail of each table row.
    let owned: usize = pages.iter().sum();
    let mut pool = owned + 3;
    while pool.is_multiple_of(7) {
        pool += 1;
    }
    let slot_of = |n: usize| (n * 7 + 3) % pool;
    let mut table = vec![slot_of(owned) as f32; BATCH * width_pages];
    let mut pool_cells = [(); 2].map(|_| vec![f32::NAN; pool * KV_HEADS * page * DIM]);
    let mut next = 0;
    for b in 0..BATCH {
        for j in 0..pages[b] {
            let slot = slot_of(next);
            next += 1;
            table[b * width_pages + j] = slot as f32;
            for (cells, source) in pool_cells.iter_mut().zip([&key, &value]) {
                for row in 0..KV_HEADS * page {
                    let (kv, t) = (row / page, row % page);
                    for d in 0..DIM {
                        cells[(slot * KV_HEADS * page + row) * DIM + d] =
                            source.get_f32(&[b, kv, j * page + t, d]);
                    }
                }
            }
        }
    }

    let scale = ScoreScale::standard(DIM).value();
    let group = HEADS / KV_HEADS;
    let mut expected = vec![0f32; BATCH * HEADS * DIM];
    for b in 0..BATCH {
        for h in 0..HEADS {
            let kv = h / group;
            let scores: Vec<f32> = (0..lengths[b])
                .map(|t| {
                    scale
                        * (0..DIM)
                            .map(|d| query.get_f32(&[b, h, 0, d]) * key.get_f32(&[b, kv, t, d]))
                            .sum::<f32>()
                })
                .collect();
            let top = scores.iter().copied().fold(f32::MIN, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = weights.iter().sum();
            for d in 0..DIM {
                // A sequence with no keys attends to nothing and writes zeros.
                expected[(b * HEADS + h) * DIM + d] = (0..lengths[b])
                    .map(|t| weights[t] / total * value.get_f32(&[b, kv, t, d]))
                    .sum();
            }
        }
    }

    let pool_shape = [pool, KV_HEADS, page, DIM];
    let [pool_key, pool_value] =
        pool_cells.map(|cells| upload(&client, &pool_shape, cells, f32_dtype));
    let table = upload(&client, &[BATCH, width_pages], table, u32_dtype);
    let lengths_tensor = upload(
        &client,
        &[BATCH],
        lengths.map(|l| l as f32).to_vec(),
        u32_dtype,
    );
    let scoring = AttentionScoring::standard(AttentionVisibility::Attended, DIM);

    let mut seen = Vec::new();
    let mut ran = 0;
    for settings in CubeShuffleAttentionRoutine.settings(&hw) {
        // One setting per distinct walk keeps the sweep short.
        let walk = (
            settings.microkernel.plane.to_string(),
            settings.microkernel.planes_along_keys.get(),
        );
        if seen.contains(&walk) {
            continue;
        }
        seen.push(walk);
        let tensors = AttentionTensors::paged(
            &client,
            PlainTensor::of_handle(query_handle.clone()),
            pool_key.clone(),
            pool_value.clone(),
            table.clone(),
            lengths_tensor.clone(),
            scoring,
        );
        let tensors = match tensors {
            Ok(tensors) => tensors,
            Err(ATTENTION_ELEMENT_UNVERIFIED) => return,
            Err(refusal) => panic!("tensors refused: {refusal:?}"),
        };
        let Ok(out) = CubeShuffleAttentionRoutine.launch_on(&hw, tensors, &settings) else {
            continue;
        };
        ran += 1;
        let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
        for at in 0..BATCH * HEADS * DIM {
            let (got, want) = (actual.data.get_f32(at), expected[at]);
            assert!(
                (got - want).abs() <= 2e-3 * (1. + want.abs()),
                "{settings}, page {page}, lengths {lengths:?}: out[{at}] = {got}, expected {want}"
            );
        }
    }
    eprintln!("page {page}, lengths {lengths:?}: {ran} launches matched the host");
    assert!(
        ran > 0 || hw.plane_dim().is_err(),
        "no setting launched on a device with a plane width, page {page}, lengths {lengths:?}"
    );
}

#[test]
fn cube_shuffle_reads_through_a_page_table() {
    for page in [16, 12, 64] {
        run([3 * page + 5, page + 1, 0, 2 * page], page);
        run([3, page - 1, 1, 0], page);
    }
}
