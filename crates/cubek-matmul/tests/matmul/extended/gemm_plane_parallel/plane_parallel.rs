// Mat × Mat (Row, Col): full GEMM, K contiguous on both sides → direct path.

fn matmat() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
}

fn matmat_row_row() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
}

fn matmat_col_col() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
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

// VecMat (m = 1): the rhs layout selects between the simple kernel (Col) and
// the staged transposed kernel (Row, CPU only).

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

#[test]
pub fn vecmat_very_small_square_row_major() {
    GemmTestCase {
        m: 1,
        n: 4,
        k: 4,
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
pub fn vecmat_k_larger_than_n_row_major() {
    GemmTestCase {
        m: 1,
        n: 128,
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
pub fn vecmat_k_smaller_than_n_row_major() {
    GemmTestCase {
        m: 1,
        n: 256,
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
pub fn vecmat_small_square_row_major() {
    GemmTestCase {
        m: 1,
        n: 256,
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
pub fn vecmat_large_row_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
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
pub fn vecmat_large_broadcast_lhs_row_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
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
pub fn vecmat_large_broadcast_rhs_row_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
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
pub fn vecmat_large_batched_row_major() {
    GemmTestCase {
        m: 1,
        n: 1280,
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
pub fn vecmat_uneven_shape_row_major() {
    GemmTestCase {
        m: 1,
        n: 32,
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
pub fn vecmat_not_same_vectorization_row_major() {
    GemmTestCase {
        m: 1,
        n: 128,
        k: 64,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

// MatVec (n = 1): the lhs layout selects between the simple kernel (Row) and
// the staged transposed kernel (Col, CPU only).

#[test]
pub fn matvec_very_small_square_col_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_k_larger_than_n_col_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_k_smaller_than_n_col_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_small_square_col_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_col_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_broadcast_lhs_col_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 1,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_broadcast_rhs_col_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_large_batched_col_major() {
    GemmTestCase {
        m: 1280,
        n: 1,
        k: 1280,
        lhs_batch: 2,
        rhs_batch: 2,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_uneven_shape_col_major() {
    GemmTestCase {
        m: 32,
        n: 1,
        k: 29,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

#[test]
pub fn matvec_not_same_vectorization_col_major() {
    GemmTestCase {
        m: 128,
        n: 1,
        k: 32,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: plane_parallel(),
    }
    .test();
}

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

// Mat × Mat (Row, Row): rhs's K axis is non-contiguous (K-stride = N), so
// rhs goes through the staged-rhs path. CPU only on plane_dim > 1 backends.
// N must be a multiple of `tile_dim`.

#[test]
pub fn matmat_very_small_square_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_k_larger_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_small_square_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_skinny_m_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_large_square_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_batched_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_broadcast_lhs_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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
pub fn matmat_broadcast_rhs_row_row() {
    let (lhs_layout, rhs_layout) = matmat_row_row();
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

// Mat × Mat (Col, Col): lhs's K axis is non-contiguous (K-stride = M), so
// lhs goes through the staged-lhs path. CPU only on plane_dim > 1 backends.
// M must be a multiple of `tile_dim`.

#[test]
pub fn matmat_very_small_square_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_k_larger_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_small_square_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_skinny_n_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_large_square_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_batched_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_broadcast_lhs_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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
pub fn matmat_broadcast_rhs_col_col() {
    let (lhs_layout, rhs_layout) = matmat_col_col();
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

// Mat × Mat (Col, Row): both operands have K non-contiguous → double-
// staged path. Each plane produces a `tile_dim × tile_dim` output block,
// so BOTH m and n must be multiples of `tile_dim`. CPU only.

fn matmat_col_row() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
}

#[test]
pub fn matmat_very_small_square_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_k_larger_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_small_square_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_large_square_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_batched_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_broadcast_lhs_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
pub fn matmat_broadcast_rhs_col_row() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
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
