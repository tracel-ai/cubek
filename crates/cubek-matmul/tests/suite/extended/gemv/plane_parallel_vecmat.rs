#[test]
pub fn test_plane_parallel_vecmat_very_small_square_col_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_k_larger_than_n_col_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_k_smaller_than_n_col_major() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_small_square_col_major() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_col_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_broadcast_lhs_col_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_broadcast_col_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_batched_col_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_uneven_shape_col_major() {
    let case = GemvTestCase {
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_not_same_vectorization_col_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 32,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_very_small_square_row_major() {
    let case = GemvTestCase {
        out_dim: 4,
        k_dim: 4,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_k_larger_than_n_row_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_k_smaller_than_n_row_major() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 128,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_small_square_row_major() {
    let case = GemvTestCase {
        out_dim: 256,
        k_dim: 256,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_row_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_broadcast_lhs_row_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 1,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_broadcast_row_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_large_batched_row_major() {
    let case = GemvTestCase {
        out_dim: 1280,
        k_dim: 1280,
        vec_batch: 2,
        mat_batch: 2,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_uneven_shape_row_major() {
    let case = GemvTestCase {
        out_dim: 32,
        k_dim: 29,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}

#[test]
pub fn test_plane_parallel_vecmat_not_same_vectorization_row_major() {
    let case = GemvTestCase {
        out_dim: 128,
        k_dim: 64,
        vec_batch: 1,
        mat_batch: 1,
        mat_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(
            GemvPlaneParallelStrategy {
                target_num_planes: 8,
            },
        )),
        kind: GemvKind::VecMat,
    }
    .test();
}
