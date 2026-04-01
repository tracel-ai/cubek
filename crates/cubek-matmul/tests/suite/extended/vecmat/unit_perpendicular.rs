use cubek_matmul::routines::vecmat_unit_perpendicular::VecMatUnitPerpendicularStrategy;

#[test]
pub fn test_unit_perpendicular_very_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 128,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_k_larger_than_n() {
    let case = VecMatTestCase {
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_k_smaller_than_n() {
    let case = VecMatTestCase {
        n: 256,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_lhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_rhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_large_broadcast_batched() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_uneven_shape() {
    let case = VecMatTestCase {
        n: 32,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_unit_perpendicular_not_same_vectorization() {
    let case = VecMatTestCase {
        n: 128,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}
