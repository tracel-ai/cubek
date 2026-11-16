//! The unit registers routine: every unit holds a block of the output in its registers. The floor
//! of every product, it serves any on any device, so it may decline none.

use cubecl::zspace::shape;
use cubek_matmul::kernels::{UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Broadcast, Declines, F16, F32, Product};
use crate::tiled::ByRoutine;

/// The default on every shape a product takes: blocks it divides and blocks it does not, a single
/// row or column, batches stated on both sides, broadcast from either, along two batch axes, and
/// read through a stride of zero.
#[test]
fn the_default_matches_the_host_on_every_shape() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::Never);
    let products = [
        // Blocks the extents divide.
        Product::of(F32).m(8).n(8).k(8).rhs_along_k(),
        Product::of(F32).m(32).n(32).k(64).rhs_along_k(),
        Product::of(F32).m(48).n(32).k(64).rhs_along_k(),
        Product::of(F32).m(64).n(64).k(128).rhs_along_k(),
        // An lhs laid along `m`, at rows no line divides, and batched.
        Product::of(F32).m(7).n(8).k(8).lhs_along_m(),
        Product::of(F32)
            .m(33)
            .n(64)
            .k(64)
            .lhs_along_m()
            .batched(shape![2], shape![2]),
        // Batched, and extents no block divides.
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .rhs_along_k()
            .batched(shape![4], shape![4]),
        Product::of(F32)
            .m(32)
            .n(48)
            .k(64)
            .rhs_along_k()
            .batched(shape![3], shape![3]),
        Product::of(F32).m(10).n(10).k(10).rhs_along_k(),
        Product::of(F32)
            .m(30)
            .n(20)
            .k(32)
            .rhs_along_k()
            .batched(shape![2], shape![2]),
        Product::of(F32).m(37).n(41).k(53).rhs_along_k(),
        // A single column, and fewer columns than a line.
        Product::of(F32).m(64).n(1).k(64),
        Product::of(F32).m(32).n(3).k(48),
        // Batches broadcast from either side, along two axes, and at extents no block divides.
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .batched(shape![4], shape![1]),
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .rhs_along_k()
            .batched(shape![1], shape![4]),
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .batched(shape![4, 1], shape![1, 3]),
        Product::of(F32)
            .m(16)
            .n(16)
            .k(32)
            .rhs_along_k()
            .batched(shape![2, 3], shape![2, 3]),
        Product::of(F32)
            .m(10)
            .n(14)
            .k(10)
            .batched(shape![3], shape![1]),
        // A batch read through a stride of zero, from either side.
        Product::of(F32)
            .m(64)
            .n(64)
            .k(128)
            .rhs_along_k()
            .batched(shape![2], shape![2])
            .broadcast(Broadcast::LhsBatch),
        Product::of(F32)
            .m(64)
            .n(64)
            .k(128)
            .rhs_along_k()
            .batched(shape![2], shape![2])
            .broadcast(Broadcast::RhsBatch),
    ];
    registers.check(registers.at([UnitRegistersMatmulRoutineSettings::DEFAULT], &products));
}

/// Every setting the routine lists, on the products of many rows and of few.
#[test]
fn every_unit_registers_setting_matches_the_host() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::Never);
    let products = [Product::many_rows(), Product::few_rows()].concat();
    registers.check(registers.at_every_setting(&products));
}

/// The settings the CPU runtime lists for a col rhs, at the depths a convolution's channels give:
/// `k` of 24 and 56, which its depth of eight divides and sixteen does not, and 272. The rows are
/// no multiple of a cube's, so the last cube is partial.
#[test]
fn the_cpu_settings_read_a_col_rhs_at_the_convolutions_depths() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::Never);
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let col_rhs = Product::of(F32).m(200).rhs_along_k();
    let products = [
        col_rhs.clone().n(48).k(24),
        col_rhs.clone().n(48).k(56),
        col_rhs.n(64).k(272),
    ];
    registers.check(registers.at(
        [UnitRegistersMatmulRoutineSettings::cpu_col_rhs(&hw)],
        &products,
    ));
}

/// Every setting sums an `i32` product exactly, in `i32`.
#[test]
fn every_unit_registers_setting_sums_i32_exactly() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::Never);
    registers.check(registers.at_every_setting(&Product::integers(32)));
}

/// `f16` inputs into an `f32` output: the sum is cast to the output's element as it drains.
#[test]
fn the_default_writes_a_wider_output() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::Never);
    let product = Product::of(F16).m(64).n(48).k(96).writing(F32);
    registers.check(registers.at_its_lead(&[product]));
}
