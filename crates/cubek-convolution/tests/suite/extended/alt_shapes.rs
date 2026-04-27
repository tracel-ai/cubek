//! Forced-blueprint tests covering awkward / non-aligned spatial sizes,
//! channel counts, and tall/skinny problems. Uses `SimpleSyncCyclicConv` as
//! the representative routine — the goal is to exercise bounds-checking and
//! vector-size handling under odd shapes, not to cover every routine.

use cubek_convolution::kernels::algorithm::simple::SimpleSyncCyclicConv;
use cubek_matmul::components::tile_matmul::cmma::CmmaMatmul;
use cubek_matmul::{components::stage::PartitionBuffering, definition::SwizzleModes};
use cubek_std::{PartitionSize, StageSize};

use super::common::{default_tile_size, f16_dtypes, tiling_scheme};
use crate::suite::launcher_strategy::{ConvolutionSize, test_algo};

fn run(size: ConvolutionSize) {
    test_algo::<SimpleSyncCyclicConv<CmmaMatmul>>(
        f16_dtypes(),
        tiling_scheme(
            default_tile_size(),
            PartitionSize { m: 1, n: 1, k: 1 },
            StageSize { m: 2, n: 2, k: 1 },
        ),
        SwizzleModes::default(),
        PartitionBuffering::Single,
        size,
    );
}

#[test]
fn shape_4x4x1x1() {
    run(ConvolutionSize {
        h: 4,
        w: 4,
        c: 1,
        out_c: 1,
    });
}

#[test]
fn shape_17x17x1x1() {
    run(ConvolutionSize {
        h: 17,
        w: 17,
        c: 1,
        out_c: 1,
    });
}

#[test]
fn shape_20x20x16x32() {
    run(ConvolutionSize {
        h: 20,
        w: 20,
        c: 16,
        out_c: 32,
    });
}

#[test]
fn shape_23x10x17x20() {
    run(ConvolutionSize {
        h: 23,
        w: 10,
        c: 17,
        out_c: 20,
    });
}

#[test]
fn shape_32x32x64x3() {
    run(ConvolutionSize {
        h: 32,
        w: 32,
        c: 64,
        out_c: 3,
    });
}
