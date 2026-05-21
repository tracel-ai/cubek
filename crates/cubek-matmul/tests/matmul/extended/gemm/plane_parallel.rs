// Mat × Mat (Row, Col): full GEMM, K contiguous on both sides.

fn matmat() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
}

#[test]
pub fn very_small_square() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 8,
        n: 8,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn k_larger() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 16,
        n: 16,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn small_square() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn skinny_m() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 4,
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn skinny_n() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 128,
        n: 4,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn large_square() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 256,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn batched() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 2,
        rhs_batch: 2,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn broadcast_lhs() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 2,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn broadcast_rhs() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 2,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn uneven_n() {
    let (lhs_layout, rhs_layout) = matmat();
    GemmTestCase {
        m: 16,
        n: 29,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

// VecMat (m = 1): rhs must be ColMajor (K-contig); lhs is a vector and is
// made contiguous by the launcher when needed.

#[test]
pub fn vecmat_very_small_square_col_major() {
    GemmTestCase {
        m: 1,
        n: 128,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_k_larger_than_n_col_major() {
    GemmTestCase {
        m: 1,
        n: 128,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_k_smaller_than_n_col_major() {
    GemmTestCase {
        m: 1,
        n: 256,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_small_square_col_major() {
    GemmTestCase {
        m: 1,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_large_col_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_large_broadcast_lhs_col_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_large_broadcast_rhs_col_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_large_batched_col_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_uneven_shape_col_major() {
    GemmTestCase {
        m: 1,
        n: 32,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn vecmat_not_same_vectorization_col_major() {
    GemmTestCase {
        m: 1,
        n: 128,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

// MatVec (n = 1): lhs must be RowMajor (K-contig); rhs is a vector and is
// made contiguous by the launcher when needed.

#[test]
pub fn matvec_very_small_square_row_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_k_larger_than_n_row_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_k_smaller_than_n_row_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_small_square_row_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_row_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_broadcast_lhs_row_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_broadcast_rhs_row_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_batched_row_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_uneven_shape_row_major() {
    GemmTestCase {
        m: 32,
        n: 1,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_not_same_vectorization_row_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}
