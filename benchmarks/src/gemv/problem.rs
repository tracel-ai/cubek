use cubecl::ir::MatrixLayout;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_VECMAT_RR: &str = "vecmat_b2_out4096_k8192_rr";
pub const PROBLEM_VECMAT_RC: &str = "vecmat_b2_out4096_k8192_rc";
pub const PROBLEM_MATVEC_RR: &str = "matvec_b2_out4096_k8192_rr";
pub const PROBLEM_MATVEC_CR: &str = "matvec_b2_out4096_k8192_cr";

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

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_VECMAT_RR.to_string(),
            label: "VecMat (b=2 out=4096 k=8192) lhs=row rhs=row".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_VECMAT_RC.to_string(),
            label: "VecMat (b=2 out=4096 k=8192) lhs=row rhs=col".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_MATVEC_RR.to_string(),
            label: "MatVec (b=2 out=4096 k=8192) lhs=row rhs=row".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_MATVEC_CR.to_string(),
            label: "MatVec (b=2 out=4096 k=8192) lhs=col rhs=row".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<GemvProblem> {
    let (batches, out_dim, k_dim) = (2, 4096, 8192);
    let (kind, lhs, rhs) = match id {
        PROBLEM_VECMAT_RR => (
            ProblemKind::VecMat,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
        ),
        PROBLEM_VECMAT_RC => (
            ProblemKind::VecMat,
            MatrixLayout::RowMajor,
            MatrixLayout::ColMajor,
        ),
        PROBLEM_MATVEC_RR => (
            ProblemKind::MatVec,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
        ),
        PROBLEM_MATVEC_CR => (
            ProblemKind::MatVec,
            MatrixLayout::ColMajor,
            MatrixLayout::RowMajor,
        ),
        _ => return None,
    };
    Some(GemvProblem {
        kind,
        batches,
        out_dim,
        k_dim,
        lhs_layout: lhs,
        rhs_layout: rhs,
    })
}
