use cubek_matmul::routines::vecmat_plane_parallel::VecMatPlaneParallelStrategy;

#[test]
pub fn test_plane_parallel_very_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 128,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_k_larger_than_n() {
    let case = VecMatTestCase {
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_k_smaller_than_n() {
    let case = VecMatTestCase {
        n: 256,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_large() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_large_broadcast_lhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_large_broadcast_rhs() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_large_broadcast_batched() {
    let case = VecMatTestCase {
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_uneven_shape() {
    let case = VecMatTestCase {
        n: 32,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}

#[test]
pub fn test_plane_parallel_not_same_vectorization() {
    let case = VecMatTestCase {
        n: 128,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}
