use cubecl::ir::MatrixLayout;

use crate::registry::CatalogEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    VecMat, // [b, 1, k] x [b, k, n] -> [b, 1, n]
    MatVec, // [b, m, k] x [b, k, 1] -> [b, m, 1]
}

pub struct GemvProblem {
    pub kind: ProblemKind,
    pub batches: usize,
    pub out_dim: usize,
    pub k_dim: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
}

pub fn problems() -> Vec<CatalogEntry<GemvProblem>> {
    let (batches, out_dim, k_dim) = (2, 4096, 8192);
    vec![
        CatalogEntry::new(
            "vecmat_b2_out4096_k8192_rr",
            "VecMat (b=2 out=4096 k=8192) lhs=row rhs=row",
            GemvProblem {
                kind: ProblemKind::VecMat,
                batches,
                out_dim,
                k_dim,
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
            },
        ),
        CatalogEntry::new(
            "vecmat_b2_out4096_k8192_rc",
            "VecMat (b=2 out=4096 k=8192) lhs=row rhs=col",
            GemvProblem {
                kind: ProblemKind::VecMat,
                batches,
                out_dim,
                k_dim,
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
            },
        ),
        CatalogEntry::new(
            "matvec_b2_out4096_k8192_rr",
            "MatVec (b=2 out=4096 k=8192) lhs=row rhs=row",
            GemvProblem {
                kind: ProblemKind::MatVec,
                batches,
                out_dim,
                k_dim,
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
            },
        ),
        CatalogEntry::new(
            "matvec_b2_out4096_k8192_cr",
            "MatVec (b=2 out=4096 k=8192) lhs=col rhs=row",
            GemvProblem {
                kind: ProblemKind::MatVec,
                batches,
                out_dim,
                k_dim,
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
            },
        ),
    ]
}
