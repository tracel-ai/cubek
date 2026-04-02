use cubek_matmul::routines::vecmat_unit_perpendicular::VecMatUnitPerpendicularStrategy;

#[test]
pub fn test_unit_perpendicular_very_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        out_dim: 128,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
        out_dim: 128,
        k_dim: 32,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatUnitPerpendicular(BlueprintStrategy::Inferred(
            VecMatUnitPerpendicularStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}
