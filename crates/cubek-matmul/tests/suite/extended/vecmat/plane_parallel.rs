use cubek_matmul::routines::vecmat_plane_parallel::VecMatPlaneParallelStrategy;

#[test]
pub fn test_plane_parallel_very_small_square_rhs() {
    let case = VecMatTestCase {
        out_dim: 128,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
pub fn test_plane_parallel_large_batched() {
    let case = VecMatTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
        out_dim: 128,
        k_dim: 32,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
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
pub fn test_plane_parallel_very_small_square_rhs_row_major() {
    let case = VecMatTestCase {
        out_dim: 128,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_k_larger_than_n_row_major() {
    let case = VecMatTestCase {
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_k_smaller_than_n_row_major() {
    let case = VecMatTestCase {
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_small_square_rhs_row_major_row_major() {
    let case = VecMatTestCase {
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_large_row_major() {
    let case = VecMatTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_large_broadcast_lhs_row_major() {
    let case = VecMatTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_large_broadcast_rhs_row_major() {
    let case = VecMatTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_large_batched_row_major() {
    let case = VecMatTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_uneven_shape_row_major() {
    let case = VecMatTestCase {
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
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
pub fn test_plane_parallel_not_same_vectorization_row_major() {
    let case = VecMatTestCase {
        out_dim: 128,
        k_dim: 32,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::VecMatPlaneParallel(BlueprintStrategy::Inferred(
            VecMatPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
    };

    test_vecmat(case);
}
