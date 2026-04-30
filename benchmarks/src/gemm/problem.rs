use cubek::std::MatrixLayout;

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_VECMAT_RR: &str = "vecmat_2x1x4096x4096_rr";
pub const PROBLEM_VECMAT_RC: &str = "vecmat_2x1x4096x4096_rc";
pub const PROBLEM_VECMAT_CR: &str = "vecmat_2x1x4096x4096_cr";
pub const PROBLEM_VECMAT_CC: &str = "vecmat_2x1x4096x4096_cc";

pub struct GemmProblem {
    pub b: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_VECMAT_RR.to_string(),
            label: "VecMat (b=2 m=1 n=4096 k=4096) row/row".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_VECMAT_RC.to_string(),
            label: "VecMat (b=2 m=1 n=4096 k=4096) row/col".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_VECMAT_CR.to_string(),
            label: "VecMat (b=2 m=1 n=4096 k=4096) col/row".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_VECMAT_CC.to_string(),
            label: "VecMat (b=2 m=1 n=4096 k=4096) col/col".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<GemmProblem> {
    let (b, m, n, k) = (2, 1, 4096, 4096);
    let (lhs, rhs) = match id {
        PROBLEM_VECMAT_RR => (MatrixLayout::RowMajor, MatrixLayout::RowMajor),
        PROBLEM_VECMAT_RC => (MatrixLayout::RowMajor, MatrixLayout::ColMajor),
        PROBLEM_VECMAT_CR => (MatrixLayout::ColMajor, MatrixLayout::RowMajor),
        PROBLEM_VECMAT_CC => (MatrixLayout::ColMajor, MatrixLayout::ColMajor),
        _ => return None,
    };
    Some(GemmProblem {
        b,
        m,
        n,
        k,
        lhs_layout: lhs,
        rhs_layout: rhs,
    })
}
