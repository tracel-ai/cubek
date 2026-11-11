//! A packed weight against a float activation.

use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
};
use cubek_test_utils::{ScopeUnderTest, TestOutcome, ValidationResult};

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::packed::{PackedProduct, Stored};

/// A packed matrix seen through a transposed view, as a framework hands a tied embedding to the
/// vocabulary head: [`PackedTensor::of_quantized`](cubek_matmul::kernels::tensor::PackedTensor::of_quantized)
/// reads it as the matrix it views, and the shuffle routines a decode step races answer its
/// product. It refused the view once, its words not row-major under it, and the framework
/// dequantized the whole head on every step.
#[test]
fn a_transposed_packed_matrix_is_read_as_the_matrix_it_views() {
    let head = PackedProduct::of(Stored::Bf16, Stored::Nvfp4ViewedTransposed { seed: 3 })
        .m(4)
        .n(96)
        .k(512);
    let plane = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    plane.check(plane.at(
        PlaneShuffleMatmulRoutine.packed_settings(),
        core::slice::from_ref(&head),
    ));
    let cube = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    cube.check(cube.at(CubeShuffleMatmulRoutine.packed_settings(cube.hw()), &[head]));
}

/// The cube shuffle's split of `K` against an NVFP4 weight, as a decode step runs a projection:
/// the weight read a scale block at a time, its scales a byte each, every plane of a cube one turn
/// of `K`, the cubes walking runs of the boxes of columns. One, three and eight rows against
/// weights whose `K` takes three to twenty-four planes, their columns a whole number of
/// eight-column boxes or not. Before a read could take two words, the words were bound a read a
/// stored word, so a read of a block stepped its rows twice as far and read past the weight.
///
/// A device it prepares on prepares every shape here: where the first is declined, a device that
/// commits to no plane width, every one is a skip; elsewhere none may be declined.
#[test]
fn the_split_k_answers_a_decode_step() {
    let decode = PackedProduct::of(Stored::Bf16, Stored::Nvfp4 { seed: 3 });
    let products = [
        decode.clone().m(1).n(64).k(2048),
        decode.clone().m(1).n(44).k(4096),
        decode.clone().m(3).n(40).k(1536),
        decode.clone().m(8).n(72).k(4096),
        decode.m(1).n(48).k(12288),
    ];
    let settings = [CubeShuffleMatmulRoutineSettings::PACKED_SPLIT_K];
    let probe = ByRoutine::new(CubeShuffleMatmulRoutine, Declines::AsItStates);
    let first = probe.at(settings, &products[..1]);
    let declines = match probe.outcome(&cubecl::test_device().client(), &first[0]) {
        TestOutcome::Validated(ValidationResult::Skipped(_)) => Declines::AsItStates,
        _ => Declines::Never,
    };
    let split = ByRoutine::new(CubeShuffleMatmulRoutine, declines);
    split.check(split.at(settings, &products));
}
