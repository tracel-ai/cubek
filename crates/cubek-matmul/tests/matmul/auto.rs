//! `Strategy::Auto`, the root's own dispatch, whichever architecture is compiled behind it: it
//! serves every product, so it may decline none.

use cubecl::features::TypeUsage;
use cubecl::prelude::*;
use cubecl::zspace::shape;
use cubek_matmul::strategy::Strategy;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Broadcast, ByStrategy, Declines, F16, F32, F64, Product};

#[test]
fn auto_matches_the_host_on_a_small_product() {
    let auto = ByStrategy::new(Strategy::Auto, Declines::Never);
    auto.check([Product::of(F16).m(16).n(16).k(16)]);
}

/// Square products in both elements, and a single row and a single column, as a decode step's
/// projections and their gradients are.
#[cfg(feature = "heavy")]
#[test]
fn auto_matches_the_host_on_medium_products() {
    let auto = ByStrategy::new(Strategy::Auto, Declines::Never);
    auto.check([
        Product::of(F16).m(256).n(256).k(256),
        Product::of(F32).m(256).n(256).k(256),
        Product::of(F16).m(1).n(256).k(256),
        Product::of(F16).m(256).n(1).k(256),
    ]);
}

/// Skipped where the device has no `f64`, as on Metal: building the inputs alone launches an
/// `f64` kernel the device refuses.
#[cfg(feature = "heavy")]
#[test]
fn auto_matches_the_host_in_f64() {
    if !f64::supported_uses(&cubecl::test_device().client()).contains(TypeUsage::Arithmetic) {
        return;
    }
    let auto = ByStrategy::new(Strategy::Auto, Declines::Never);
    auto.check([Product::of(F64).m(256).n(256).k(256)]);
}

/// Batches read through a stride of zero, from either side; and the reported product: an lhs
/// `[300, 4, 256]` whose rows broadcast (strides `[256, 0, 1]`) against one col rhs
/// `[1, 256, 256]`, which `Auto` makes contiguous before it runs.
#[cfg(feature = "extended")]
#[test]
fn auto_reads_through_a_stride_of_zero() {
    let auto = ByStrategy::new(Strategy::Auto, Declines::Never);
    let batched = Product::of(F32)
        .m(64)
        .n(64)
        .k(128)
        .rhs_along_k()
        .batched(shape![2], shape![2]);
    auto.check([
        batched.clone().broadcast(Broadcast::LhsBatch),
        batched.broadcast(Broadcast::RhsBatch),
        Product::of(F32)
            .m(4)
            .n(256)
            .k(256)
            .rhs_along_k()
            .batched(shape![300], shape![1])
            .broadcast(Broadcast::LhsRows),
    ]);
}

/// A weight stored in tiles is a fact of the data: `Auto` sends it to the tiled cmma routine, which
/// stages to its tiles, whichever architecture it would have picked for the shape.
#[cfg(feature = "tiled")]
#[test]
fn auto_sends_a_weight_stored_in_tiles_to_the_tiled_cmma() {
    use crate::matmul::RhsTiles;
    let auto = ByStrategy::new(Strategy::Auto, Declines::Never);
    auto.check([Product::of(F32)
        .m(64)
        .n(512)
        .k(256)
        .rhs_in(RhsTiles { n: 64, k: 32 })]);
}
