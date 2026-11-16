//! The tile-DSL strategies a caller of [`launch_ref`](cubek_matmul::launch::launch_ref) names, as
//! burn reaches them: `Cmma` runs the cube staged routine, `CpuGemm` the unit registers one.

use cubecl::zspace::shape;
use cubek_matmul::kernels::{CubeStagedMatmulRoutineSettings, UnitRegistersMatmulRoutineSettings};
use cubek_matmul::tiled::Strategy as Tiled;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{ByStrategy, Declines, F16, F32, Product, RhsTiles};

/// `Cmma` where the device offers a matrix instruction: square in both elements, a rectangle,
/// batches, an lhs laid along `m`, a weight stored in tiles, and a wider output.
#[test]
fn cmma_matches_the_host() {
    let cmma = ByStrategy::new(
        Tiled::Cmma(CubeStagedMatmulRoutineSettings::DEFAULT),
        Declines::AsItStates,
    );
    cmma.check([
        Product::of(F32).m(256).n(256).k(256),
        Product::of(F16).m(256).n(256).k(256),
        Product::of(F32).m(64).n(128).k(32),
        Product::of(F32)
            .m(64)
            .n(64)
            .k(64)
            .batched(shape![3], shape![3]),
        Product::of(F16).m(256).n(128).k(64).lhs_along_m(),
        Product::of(F16)
            .m(128)
            .n(256)
            .k(384)
            .rhs_in(RhsTiles { n: 32, k: 32 }),
        Product::of(F16).m(64).n(48).k(96).writing(F32),
    ]);
}

/// `CpuGemm` on any device: a wider output, and a batch the rhs broadcasts.
#[test]
fn cpu_gemm_matches_the_host() {
    let cpu_gemm = ByStrategy::new(
        Tiled::CpuGemm(UnitRegistersMatmulRoutineSettings::DEFAULT),
        Declines::Never,
    );
    cpu_gemm.check([
        Product::of(F16).m(64).n(48).k(96).writing(F32),
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .batched(shape![4], shape![1]),
    ]);
}
