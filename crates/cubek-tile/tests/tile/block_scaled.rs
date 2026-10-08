//! `acc.mma(&a.mul(&sa), &b.mul(&sb))` over `e2m1` values under `e4m3` scales, one every sixteen
//! values of `k`: NVFP4, contracted through the device's block-scaled instruction where it offers
//! one, the values and scales handed to it as they lie rather than decoded first.
//!
//! The kernel is the plain staged walk a float product takes; nothing in it names the
//! instruction. Whether the block-scaled path ran is the device's to say, so the test also holds
//! the device that offers it to the reference on values the decoding path would read the same:
//! codes and scales that are exact in `f32`.

use cubecl::{client::Client, ir::FloatKind, prelude::*, quant::scheme::QuantValue, zspace::Shape};
use cubecl_common::e2m1;
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::{
    Accumulate, AccumulateExpand, Axis, Instruction, Level, Levels, Partitioning, Projection,
    Semiring, Space, StageStorage, Stages, TileArg, TileArgLaunch, TileSpec, kind::Field,
    launch::scale_tile, layout::PhysicalAxisMap, ops::matmul::MmaIo,
};

use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Values one scale covers along `k`: NVFP4's block.
const BLOCK: usize = 16;

/// `c = (a ⊗ sa) · (b ⊗ sb)`, a stage of `K` at a time, contracted through the manual-mma
/// instruction.
#[cube(launch)]
fn block_scaled_matmul<E: Numeric>(
    a: &TileArg<'_, u32, Const<1>>,
    a_scales: &TileArg<'_, f32, Const<1>>,
    b: &TileArg<'_, u32, Const<1>>,
    b_scales: &TileArg<'_, f32, Const<1>>,
    a_global: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    b_global: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile_as::<E>(comptime!(space.clone()));
    let a_scales = a_scales.tile(comptime!(space.clone()));
    let b = b.tile_as::<E>(comptime!(space.clone()));
    let b_scales = b_scales.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.accumulator::<E, E, E>(
        &a,
        &b,
        comptime!(Instruction::Mma {
            io: MmaIo::manual()
        }),
        Semiring::SUM_PROD,
    );
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        let a_scales_r = a_scales.at(region);
        let b_scales_r = b_scales.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(&a_s.mul(&a_scales_r), &b_s.mul(&b_scales_r));
        });
    });
    // The per-tensor factors above the block scales, which the instruction holds no level for.
    acc.scale(&scale_tile::<E>(a_global, comptime!(space.clone())));
    acc.scale(&scale_tile::<E>(b_global, comptime!(space.clone())));
    acc.drained_into(&c);
}

/// A `16 × 8` product of `k` deep, `a` row-major and `b` held `[n, k]`, against the host sum of
/// the decoded values under their scales, and under each operand's per-tensor factor where
/// `globals` states them. Skips where the device offers no block-scaled `e2m1` instruction, whose
/// path is the subject here.
fn check_block_scaled(k: usize, globals: Option<(f32, f32)>) {
    let client = cubecl::test_device().client();
    if !offers_nvfp4(&client) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device offers no block-scaled e2m1 instruction under e4m3 scales".to_string(),
        ))
        .enforce();
        return;
    }
    let (m, n) = (16, 8);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, 64)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    // Every code of `e2m1`, varied by position so no row or column repeats another.
    let a_codes: Vec<u8> = (0..m * k).map(|i| ((i * 7 + i / k) % 16) as u8).collect();
    let b_codes: Vec<u8> = (0..n * k)
        .map(|i| ((i * 5 + 3 * (i / k)) % 16) as u8)
        .collect();
    // Scales exact in `e4m3`: a power of two times one of its mantissas.
    let scale = |i: usize| (1 + i % 8) as f32 * 2f32.powi((i % 5) as i32 - 3);
    let a_scale_values: Vec<f32> = (0..m * k / BLOCK).map(scale).collect();
    let b_scale_values: Vec<f32> = (0..n * k / BLOCK).map(|i| scale(i + 3)).collect();

    let words = |codes: &[u8]| -> Vec<u32> {
        codes
            .chunks(8)
            .map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |acc, (j, &code)| acc | (code as u32) << (4 * j))
            })
            .collect()
    };
    let a_words = client.create_from_slice(u32::as_bytes(&words(&a_codes)));
    let b_words = client.create_from_slice(u32::as_bytes(&words(&b_codes)));
    let a_scales = client.create_from_slice(f32::as_bytes(&a_scale_values));
    let b_scales = client.create_from_slice(f32::as_bytes(&b_scale_values));
    let c = TestInput::builder(client.clone(), Shape::from(vec![m, n]))
        .zeros()
        .generate_without_host_data();

    let field = Field::Quant(QuantValue::E2M1);
    // Values counted in values; their buffers hold words of eight along `k`.
    let values = |rows: Axis| TileSpec::new(Projection::direct(&[rows, K])).packed(field);
    let scales = |rows: Axis| {
        TileSpec::new(Projection::new(
            &[rows, K],
            &[
                PhysicalAxisMap::of(rows),
                PhysicalAxisMap::of(K).over(BLOCK),
            ],
        ))
    };
    let tensor = |handle, rows: usize, cols: usize| unsafe {
        TensorArg::from_raw_parts(handle, [cols, 1].into(), [rows, cols].into())
    };
    // One `f32` over the whole operand, read as the word it is stored in.
    let global = |value: f32| {
        let word = client.create_from_slice(f32::as_bytes(&[value]));
        let spec = TileSpec::new(Projection::new(&[M, N, K], &[PhysicalAxisMap::broadcast()]))
            .packed(Field::Float(FloatKind::F32));
        ComptimeOptionArgs::Some(TileArgLaunch::new(tensor(word, 1, 1), spec))
    };
    let (a_global, b_global) = match globals {
        Some((a, b)) => (global(a), global(b)),
        None => (ComptimeOptionArgs::None, ComptimeOptionArgs::None),
    };
    block_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(tensor(a_words, m, k), values(M)),
        TileArgLaunch::new(tensor(a_scales, m, k / BLOCK), scales(M)),
        TileArgLaunch::new(tensor(b_words, n, k), values(N)),
        TileArgLaunch::new(tensor(b_scales, n, k / BLOCK), scales(N)),
        a_global,
        b_global,
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32::elem_type_native(),
    );

    let decoded = |code: u8| e2m1::from_bits(code).to_f32();
    let (a_global, b_global) = globals.unwrap_or((1.0, 1.0));
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k)
                .map(|l| {
                    decoded(a_codes[i * k + l])
                        * a_scale_values[i * (k / BLOCK) + l / BLOCK]
                        * decoded(b_codes[j * k + l])
                        * b_scale_values[j * (k / BLOCK) + l / BLOCK]
                })
                .sum::<f32>()
                * a_global
                * b_global;
            let have = got.get_f32(&[i, j]);
            assert!(
                (have - want).abs() <= 1e-3 * want.abs().max(1.0),
                "k = {k}, element ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// Whether `client`'s device offers the block-scaled `16 × 8 × 64` instruction over `e2m1`
/// operands under `e4m3` scales.
pub(crate) fn offers_nvfp4(client: &Client) -> bool {
    let fp4 = ElemType::Float(FloatKind::E2M1x2);
    client
        .properties()
        .features
        .matmul
        .scaled_mma
        .iter()
        .any(|config| {
            config.a_type == fp4
                && config.b_type == fp4
                && config.scales_type == ElemType::Float(FloatKind::E4M3)
                && (config.m, config.n, config.k, config.scales_factor) == (16, 8, 64, 4)
        })
}

/// One instruction's depth: a single stage, a single step.
#[test]
fn one_block_scaled_step_matches_the_reference() {
    check_block_scaled(64, None);
}

/// Several stages of `K`, each a step of the instruction, summed into one accumulator.
#[test]
fn a_walk_of_block_scaled_steps_matches_the_reference() {
    check_block_scaled(256, None);
}

/// NVFP4's second level: a factor over each whole operand, which the instruction holds no level
/// for, multiplied into the sum before it drains.
#[test]
fn per_tensor_factors_scale_the_block_scaled_sum() {
    check_block_scaled(128, Some((0.375, 1.0 / 448.0)));
}
