//! A packed weight against a float activation.

use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine,
    CubeStagedTmaFillerMatmulRoutine, HypercubeLastArrivalMatmulRoutine, LoadingSchedule,
    PlaneShuffleMatmulRoutine,
};
use cubek_test_utils::{ScopeUnderTest, TestOutcome, ValidationResult};

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedFormat, PackedProduct, Stored};

/// A packed matrix seen through a transposed view, as a framework hands a tied embedding to the
/// vocabulary head: [`PackedTensor::of_quantized`](cubek_matmul::kernels::tensor::PackedTensor::of_quantized)
/// reads it as the matrix it views, and the shuffle routines a decode step races answer its
/// product. It refused the view once, its words not row-major under it, and the framework
/// dequantized the whole head on every step.
#[test]
fn a_transposed_packed_matrix_is_read_as_the_matrix_it_views() {
    let head = PackedProduct::of(
        Stored::Bf16,
        Stored::PackedViewedTransposed {
            format: PackedFormat::Nvfp4,
            seed: 3,
        },
    )
    .m(4)
    .n(96)
    .k(512);
    let plane = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    plane.check_some(plane.at(
        PlaneShuffleMatmulRoutine::packed_settings(),
        core::slice::from_ref(&head),
    ));
    let cube = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    cube.check(cube.at(
        CubeShuffleMatmulRoutine::packed_settings(cube.hw()),
        &[head],
    ));
}

/// The cube shuffle's split of `K` against an NVFP4 weight, as a decode step runs a projection:
/// the weight read a scale block at a time, its scales a byte each, every plane of a cube one turn
/// of `K`, the cubes walking runs of the boxes of columns. One, three and eight rows against
/// weights whose `K` takes three to twenty-four planes, their columns a whole number of
/// eight-column boxes or not, the planes taking `K` in runs or in turns
/// ([`LoadingSchedule::ONE_TRANSACTION`]), where a turn the planes do not fill is the last. Before
/// a read could take two words, the words were bound a read a stored word, so a read of a block
/// stepped its rows twice as far and read past the weight.
///
/// A device it prepares on prepares every shape here: where the first is declined, a device that
/// commits to no plane width, every one is a skip; elsewhere none may be declined.
#[test]
fn the_split_k_answers_a_decode_step() {
    let decode = PackedProduct::of(
        Stored::Bf16,
        Stored::Packed {
            format: PackedFormat::Nvfp4,
            seed: 3,
        },
    );
    let products = [
        decode.clone().m(1).n(64).k(2048),
        decode.clone().m(1).n(44).k(4096),
        decode.clone().m(3).n(40).k(1536),
        decode.clone().m(8).n(72).k(4096),
        decode.m(1).n(48).k(12288),
    ];
    let settings = [
        LoadingSchedule::SPLIT_K.settings(),
        LoadingSchedule::ONE_TRANSACTION.settings(),
    ];
    let probe = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    let first = probe.at(settings, &products[..1]);
    let declines = match probe.outcome(&cubecl::test_device().client(), &first[0]) {
        TestOutcome::Validated(ValidationResult::Skipped(_)) => Declines::AsItStates,
        _ => Declines::Never,
    };
    let split = ByRoutine::new(CubeShuffleMatmulRoutine, declines);
    split.check(split.at(settings, &products));
}

/// A packed weight against a `bf16` activation at a prompt's rows, as a 4-bit model's prefill runs
/// a projection, at every packed setting of the routines staging it, the units', the copy
/// engine's, the TMA filler's and the hypercube's: NVFP4, MXFP4 and MXFP8, in rows and in the tile
/// order a load stores a weight in on the device, at a whole box and at rows and columns no box
/// divides. A weight in tile order once bound its scales a word of four blocks a load beside a
/// window of two, which every kernel landing it by TMA panicked expanding. Every box reads the weight as widely as its stage, its window and its units' runs
/// hold whole: MXFP4's four 32-wide blocks once staged 128 of `K` a step, and on an RTX PRO 6000
/// the boxes leading NVFP4 outgrew shared memory on MXFP4 alone; the copy engine's boxes staged
/// as stored once panicked compiling on MXFP4, their window of 64 holding half a word of its
/// scales, and its 32-value loads longer than a unit's run of 16.
#[test]
fn a_packed_weight_answers_a_prompt_at_every_staged_setting() {
    let products: Vec<PackedProduct> = [
        PackedFormat::Nvfp4,
        PackedFormat::Mxfp4,
        PackedFormat::Mxfp8,
    ]
    .into_iter()
    .flat_map(|format| {
        [
            Stored::Packed { format, seed: 5 },
            Stored::PackedInTileOrder { format, seed: 5 },
        ]
        .into_iter()
        .flat_map(|weight| {
            let projection = PackedProduct::of(Stored::Bf16, weight);
            [
                projection.clone().m(128).n(128).k(2048),
                projection.m(33).n(96).k(1024),
            ]
        })
    })
    .collect();
    let client = cubecl::test_device().client();
    // A device that runs no plane of these settings offers none or declines them, which skips;
    // one that runs them must launch at least one.
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let cases = staged.at(
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        &products,
    );
    match cases.first().map(|first| staged.outcome(&client, first)) {
        Some(TestOutcome::Validated(ValidationResult::Skipped(_))) | None => staged.check(cases),
        Some(_) => staged.check_some(cases),
    }
    let async_copy = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    async_copy.check(async_copy.at(
        CubeStagedAsyncCopyMatmulRoutine.packed_settings(async_copy.hw()),
        &products,
    ));
    let filler = ByRoutine::new(CubeStagedTmaFillerMatmulRoutine, Declines::AsItStates);
    filler.check(filler.at(
        CubeStagedTmaFillerMatmulRoutine.packed_settings(filler.hw()),
        &products,
    ));
    let hypercube = ByRoutine::new(HypercubeLastArrivalMatmulRoutine, Declines::AsItStates);
    let cases = hypercube.at(
        HypercubeLastArrivalMatmulRoutine.packed_settings(hypercube.hw()),
        &products,
    );
    match cases.first().map(|first| hypercube.outcome(&client, first)) {
        Some(TestOutcome::Validated(ValidationResult::Skipped(_))) | None => hypercube.check(cases),
        Some(_) => hypercube.check_some(cases),
    }
}

/// A packed weight in the tile order a load stores it in, landed by the TMA engine beside the
/// float activation the filling planes keep, at a prompt's rows against a model's depth of `K`:
/// NVFP4 and MXFP4, the formats whose tiles a box lands (an MXFP8 tile is not). MXFP4's window of
/// 64 holds half a word of its scales, so it reads them one at a time, a byte apiece, which the
/// units load beside the box: the copy engine was once handed them and panicked expanding every
/// such kernel.
///
/// A device whose engine lands no NVFP4 weight skips; elsewhere each format launches at least
/// one setting.
#[test]
fn a_packed_weight_in_tile_order_lands_by_tma_at_a_model_s_depth() {
    let client = cubecl::test_device().client();
    let filler = ByRoutine::new(CubeStagedTmaFillerMatmulRoutine, Declines::AsItStates);
    let at_depth = |format| {
        let weight = Stored::PackedInTileOrder { format, seed: 7 };
        filler.at(
            CubeStagedTmaFillerMatmulRoutine.packed_settings(filler.hw()),
            &[PackedProduct::of(Stored::Bf16, weight).m(96).n(512).k(5120)],
        )
    };
    let nvfp4 = at_depth(PackedFormat::Nvfp4);
    let lands = nvfp4.iter().any(|case| {
        !matches!(
            filler.outcome(&client, case),
            TestOutcome::Validated(ValidationResult::Skipped(_))
        )
    });
    for cases in [nvfp4, at_depth(PackedFormat::Mxfp4)] {
        match lands {
            true => filler.check_some(cases),
            false => filler.check(cases),
        }
    }
}
