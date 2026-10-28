//! Inferred-blueprint smoke tests for TMA routines.

use cubek_matmul::multi_level::Strategy as MultiLevel;

use crate::harness::{client, f16_elems, rect, square, test_matmul_strategy};

#[test]
fn simple_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTmaCmma(Default::default()).into(),
    );
}

#[test]
fn simple_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SimpleTmaMma(Default::default()).into(),
    );
}

#[test]
fn specialized_tma_cmma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedTmaCmma(Default::default()).into(),
    );
}

#[test]
fn specialized_tma_mma() {
    test_matmul_strategy(
        client(),
        square(256, f16_elems()),
        MultiLevel::SpecializedTmaMma(Default::default()).into(),
    );
}

/// A problem four times taller than it is wide: the cmma instruction is `32x8x16` there, and the
/// inferred stage is sixteen of them high, 512 rows, past the 256 elements a tensor map's box can
/// span. The routine has to refuse that stage when it is set up: launched, its tensor map fails to
/// encode, and nothing is written.
#[test]
fn simple_tma_cmma_tall() {
    test_matmul_strategy(
        client(),
        rect(1024, 64, 256, f16_elems()),
        MultiLevel::SimpleTmaCmma(Default::default()).into(),
    );
}

/// [`simple_tma_cmma_tall`] for the specialized routine, which infers the same stage.
#[test]
fn specialized_tma_cmma_tall() {
    test_matmul_strategy(
        client(),
        rect(1024, 64, 256, f16_elems()),
        MultiLevel::SpecializedTmaCmma(Default::default()).into(),
    );
}
