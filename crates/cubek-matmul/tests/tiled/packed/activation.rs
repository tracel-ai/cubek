//! A packed activation, as W4A4 runs a projection: against a float weight, and against a packed
//! one, through the cube staged routines that stage a packed input for a matrix instruction.

use cubek_matmul::kernels::{
    CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings,
    CubeStagedTmaFillerMatmulRoutine, CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
    PlaneShuffleMatmulRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedFormat, PackedProduct, Stored};

/// NVFP4 activations against a `bf16` weight: a gated-delta projection 48 columns wide at a
/// prompt's rows, and the same 44 wide. A box 32 columns wide reaches past the weight's columns,
/// which binds it at a width dividing them; on an RTX PRO 6000 a box bound eight wide past 48,
/// which no mask stated, refused to bind and the model did not load.
#[test]
fn a_packed_activation_against_a_float_weight_answers_at_every_setting() {
    let projection = PackedProduct::of(
        Stored::Packed {
            format: PackedFormat::Nvfp4,
            seed: 1,
        },
        Stored::Bf16,
    )
    .m(512)
    .k(5120);
    let products = [projection.clone().n(48), projection.n(44)];
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        CubeStagedMatmulRoutine.packed_one_step_settings(),
    ]
    .concat();
    staged.check(staged.at(settings, &products));
}

/// Activations against a weight in the same block-scaled format, NVFP4, MXFP4 and MXFP8, a
/// model's projections at a prompt's rows, and at rows and columns no box divides, a prompt's
/// ragged last chunk: every box past the matrix's edge reads zeros and writes nothing. On an RTX
/// PRO 6000 every setting whose planes hold a grid of block-scaled fragments, rather than one,
/// answered zero: the grid's fragments were loaded into copies and contracted unwritten, and the
/// model's prompt was lost. MXFP4 and MXFP8 once reached no block-scaled instruction: a step of
/// it held part of a word of their scales, which they are now read one at a time for.
#[test]
fn a_packed_activation_against_a_packed_weight_answers_at_every_setting() {
    let products: Vec<PackedProduct> = [
        PackedFormat::Nvfp4,
        PackedFormat::Mxfp4,
        PackedFormat::Mxfp8,
    ]
    .into_iter()
    .flat_map(|format| {
        let both = PackedProduct::of(
            Stored::Packed { format, seed: 1 },
            Stored::Packed { format, seed: 2 },
        )
        .k(5120);
        [
            both.clone().m(512).n(128),
            both.clone().m(33).n(96),
            both.m(300).n(96),
        ]
    })
    .collect();
    let products = products.as_slice();

    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        CubeStagedMatmulRoutine.packed_one_step_settings(),
        vec![CubeStagedMatmulRoutineSettings::DEFAULT],
    ]
    .concat();
    staged.check(staged.at(settings, products));

    let async_copy = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    let settings = [
        CubeStagedAsyncCopyMatmulRoutine.packed_settings(async_copy.hw()),
        CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(async_copy.hw()),
    ]
    .concat();
    async_copy.check(async_copy.at(settings, products));

    let filler = ByRoutine::new(CubeStagedTmaFillerMatmulRoutine, Declines::AsItStates);
    filler.check(filler.at(
        CubeStagedTmaFillerMatmulRoutine.packed_settings(filler.hw()),
        products,
    ));

    let tma = ByRoutine::new(CubeStagedTmaMatmulRoutine, Declines::AsItStates);
    tma.check(tma.at(
        CubeStagedTmaMatmulRoutine.block_scaled_settings(tma.hw()),
        products,
    ));

    let stored = ByRoutine::new(CubeStagedTmaStoredMatmulRoutine, Declines::AsItStates);
    stored.check(stored.at(
        CubeStagedTmaStoredMatmulRoutine.block_scaled_settings(stored.hw()),
        products,
    ));
}

/// MXFP8 activations against an MXFP4 weight, W4A8 as a prompt runs it, a model's projections
/// and rows and columns no box divides: every block-scaled setting of the routines that read both
/// inputs as stored. Both share `ue8m0` scales every thirty-two, so a device whose 8-bit float
/// instruction multiplies them takes the weight's `e2m1` codes widened to `e4m3`, exactly.
#[test]
fn an_mxfp8_activation_against_an_mxfp4_weight_answers_at_every_setting() {
    let w4a8 = PackedProduct::of(
        Stored::Packed {
            format: PackedFormat::Mxfp8,
            seed: 1,
        },
        Stored::Packed {
            format: PackedFormat::Mxfp4,
            seed: 2,
        },
    )
    .k(5120);
    let products = [
        w4a8.clone().m(512).n(128),
        w4a8.clone().m(33).n(96),
        w4a8.m(300).n(96),
    ];

    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    staged.check(staged.at(vec![CubeStagedMatmulRoutineSettings::DEFAULT], &products));

    let async_copy = ByRoutine::new(CubeStagedAsyncCopyMatmulRoutine, Declines::AsItStates);
    async_copy.check(async_copy.at(
        CubeStagedAsyncCopyMatmulRoutine.block_scaled_settings(async_copy.hw()),
        &products,
    ));
}

/// Packed activations against a packed weight of another width, as a decode step mixing formats
/// runs: every packed setting of the plane shuffle. A `q8` activation against a `q4` weight is lined
/// at the eight values a step of the weight serves, two words of the activation; a `q4`
/// activation's word of eight against a `q8` weight's step of four is declined
/// ([`PlaneShuffleMicrokernel::PACKED_LHS_READ_PAST_RHS`](cubek_microkernel::matmul::plane::PlaneShuffleMicrokernel::PACKED_LHS_READ_PAST_RHS)):
/// on an RTX PRO 6000 it once panicked compiling every setting.
#[test]
fn packed_inputs_of_two_widths_answer_or_are_declined_by_the_plane_shuffle() {
    let packed = |format, seed| Stored::Packed { format, seed };
    let product = |lhs, rhs| {
        PackedProduct::of(packed(lhs, 1), packed(rhs, 2))
            .m(8)
            .n(64)
            .k(1024)
    };
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check_some(shuffle.at(
        PlaneShuffleMatmulRoutine::packed_settings(),
        &[product(PackedFormat::Q8, PackedFormat::Q4)],
    ));
    shuffle.check(shuffle.at(
        PlaneShuffleMatmulRoutine::packed_settings(),
        &[product(PackedFormat::Q4, PackedFormat::Q8)],
    ));
}
