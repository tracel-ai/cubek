//! A wide `ArgTopK` on the cube kernel, whose planes stage `k` values and indices each in shared
//! memory: at `k = 13` over a 4096-long row, a cube sized from the row alone would ask more shared
//! memory than a device has, so the kernel must size its cube to fit. Gated behind `heavy` like
//! the other cube tests, whose kernels are slow to compile.

#![cfg(feature = "heavy")]

use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::ScopeChecker;

use crate::reduce::it::test_case::{CUBE, Op, ReduceScopeChecker, TestCase};

/// `ArgTopK(13)` over `[64, 4096]` (axis 1) on the cube kernel, against the CPU top-k. A single
/// representative `k` keeps the suite fast, since a cube kernel is slow to compile whatever `k` it
/// is built for.
#[test]
fn argtopk_k13_on_the_cube_launches() {
    ReduceScopeChecker.check([TestCase::new::<f32>(
        Shape::new([64, 4096]),
        Strides::new(&[4096, 1]),
        Some(1),
        CUBE,
    )
    .reduced_by(Op::ArgTopK(13))]);
}
