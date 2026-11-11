//! A packed weight against a float activation.

use cubek_matmul::kernels::{CubeShuffleMatmulRoutine, PlaneShuffleMatmulRoutine};
use cubek_test_utils::ScopeUnderTest;

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
