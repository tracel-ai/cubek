#[test]
pub fn very_small_square() {
    GemmTestCase {
        m: 8,
        n: 8,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn k_larger() {
    GemmTestCase {
        m: 16,
        n: 16,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn small_square() {
    GemmTestCase {
        m: 32,
        n: 32,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn skinny_m() {
    GemmTestCase {
        m: 4,
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn skinny_n() {
    GemmTestCase {
        m: 128,
        n: 4,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn large_square() {
    GemmTestCase {
        m: 256,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn batched() {
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 2,
        rhs_batch: 2,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn broadcast_lhs() {
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 2,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn broadcast_rhs() {
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 2,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}

#[test]
pub fn uneven_n() {
    GemmTestCase {
        m: 16,
        n: 29,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        elems: elems(),
        strategy: Strategy::GemmPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
    }
    .test();
}
