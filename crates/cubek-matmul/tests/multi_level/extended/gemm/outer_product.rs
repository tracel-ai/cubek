// All sizes here are chosen to be divisible by the largest vector_size the
// CPU backend picks (8 for f32, 16 for f16). Outer-product variants need
// the block axis (N for OuterN, M for OuterM) divisible by vector_size;
// the Routine validates this.

// =====================================================================
// Mat × Mat (Row, Col) — Variant::Dot (same kernel the plane-parallel cases hit).
// =====================================================================

fn matmat_row_col() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
}

#[test]
pub fn matmat_small_square_row_col() {
    let (lhs_layout, rhs_layout) = matmat_row_col();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn matmat_large_square_row_col() {
    let (lhs_layout, rhs_layout) = matmat_row_col();
    GemmTestCase {
        m: 256,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn matmat_batched_row_col() {
    let (lhs_layout, rhs_layout) = matmat_row_col();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 2,
        rhs_batch: 2,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn matmat_broadcast_lhs_row_col() {
    let (lhs_layout, rhs_layout) = matmat_row_col();
    GemmTestCase {
        m: 32,
        n: 32,
        k: 128,
        lhs_batch: 1,
        rhs_batch: 2,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
    }
    .test();
}

// =====================================================================
// Mat × Mat (Row, Row) — Variant::OuterNLhsContig.
// =====================================================================

fn matmat_row_row() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
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
        strategy: outer_product(),
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
        strategy: outer_product(),
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
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

// =====================================================================
// Mat × Mat (Col, Col) — Variant::OuterM.
// =====================================================================

fn matmat_col_col() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::ColMajor, MatrixLayout::ColMajor)
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
        strategy: outer_product(),
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
        strategy: outer_product(),
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
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

// =====================================================================
// Mat × Mat (Col, Row) — no variant reads this layout pair: `launch_ref`
// normalizes it to Dot where planes exist, and the routine rejects it on
// CPU (the `cpu_gemm` routine serves it there). Kept as coverage that
// both paths give the right answer, whichever one runs.
// =====================================================================

fn matmat_col_row() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::ColMajor, MatrixLayout::RowMajor)
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
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

/// A transposed lhs whose M is not a multiple of the vector size — the
/// `a.transpose().matmul(b)` shape from burn#5304. The Col-Row variant read
/// the lhs as M-vectors and silently returned another row's data; no variant
/// claims the layout now.
#[test]
pub fn matmat_col_row_m_not_vector_multiple() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
    GemmTestCase {
        m: 7,
        n: 8,
        k: 8,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn matmat_large_col_row_m_not_vector_multiple() {
    let (lhs_layout, rhs_layout) = matmat_col_row();
    GemmTestCase {
        m: 33,
        n: 64,
        k: 64,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout,
        rhs_layout,
        elems: elems(),
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

// =====================================================================
// VecMat (m = 1)
// =====================================================================

#[test]
pub fn vecmat_col_major() {
    GemmTestCase {
        m: 1,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        elems: elems(),
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn vecmat_row_major() {
    GemmTestCase {
        m: 1,
        n: 256,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

// =====================================================================
// MatVec (n = 1)
// =====================================================================

#[test]
pub fn matvec_row_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

#[test]
pub fn matvec_col_major() {
    GemmTestCase {
        m: 256,
        n: 1,
        k: 256,
        lhs_batch: 1,
        rhs_batch: 1,
        lhs_layout: MatrixLayout::ColMajor,
        rhs_layout: MatrixLayout::RowMajor,
        elems: elems(),
        strategy: outer_product(),
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
        strategy: outer_product(),
    }
    .test();
}

// Assert these layouts actually execute on GPU instead of being silently rejected.

#[test]
pub fn matmat_row_row_executes() {
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
        strategy: outer_product(),
    }
    .test_executes();
}

#[test]
pub fn matmat_col_col_executes() {
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
        strategy: outer_product(),
    }
    .test_executes();
}

#[test]
pub fn matmat_col_row_executes() {
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
        strategy: outer_product(),
    }
    .test_executes();
}
