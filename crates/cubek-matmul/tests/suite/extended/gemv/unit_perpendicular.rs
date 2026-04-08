#[test]
pub fn test_unit_perpendicular_very_small_square_rhs_row_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_k_larger_than_n() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_k_smaller_than_n() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_small_square_rhs_row_major() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_large() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_lhs() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_rhs() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_large_batched() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_uneven_shape() {
    let case = GemvTestCase {
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_unit_perpendicular_not_same_vectorization() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 32,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(
            GemvUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}
