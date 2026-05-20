use cubek_std::MatrixLayout;

use crate::{
    components::{
        batch::{BatchConfig, CheckBounds},
        global::memory::GlobalLayoutConfig,
    },
    definition::{MatmulProblem, MatmulSetupError},
};

/// Per-operand layout classification, with a special case for vectors.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum OperandLayout {
    RowMajor,
    ColMajor,
    Vector,
}

/// Layouts of both operands as derived from the problem.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MatmulOperandLayouts {
    pub lhs: OperandLayout,
    pub rhs: OperandLayout,
}

impl MatmulOperandLayouts {
    pub(crate) fn from_problem(
        problem: &MatmulProblem,
    ) -> Result<MatmulOperandLayouts, MatmulSetupError> {
        let lhs = operand_kind(problem.m, problem.lhs_layout);
        let rhs = operand_kind(problem.n, problem.rhs_layout);
        Ok(MatmulOperandLayouts { lhs, rhs })
    }

    /// Pick the kernel variant appropriate for these layouts.
    ///
    /// `Dot` (Row-Col): K is contiguous on both sides — dot-product
    /// reduction along K, one cell per plane. Supports `plane_dim > 1`
    /// (units cooperate across K).
    ///
    /// `OuterNLhsContig` (Row-Row): rhs is N-contig (RowMajor) and lhs
    /// is K-contig (RowMajor or vector) — outer-product vectorized along
    /// N with a single LHS K-vector load. CPU-only (`plane_dim == 1`).
    ///
    /// `OuterNLhsStrided` (Col-Row): rhs is N-contig but lhs is M-contig
    /// (ColMajor) — outer-product vectorized along N with scalar LHS
    /// reads (strided in K). CPU-only.
    ///
    /// `OuterM` (Col-Col): lhs is M-contig (ColMajor) and rhs is K-contig
    /// (ColMajor or vector) — outer-product vectorized along M with a
    /// single RHS K-vector load. CPU-only.
    pub fn variant(self) -> Variant {
        use OperandLayout::*;
        let lhs_k_contig = matches!(self.lhs, RowMajor | Vector);
        let rhs_k_contig = matches!(self.rhs, ColMajor | Vector);
        let rhs_n_contig = matches!(self.rhs, RowMajor);
        let lhs_m_contig = matches!(self.lhs, ColMajor);
        match (lhs_k_contig, rhs_k_contig, rhs_n_contig, lhs_m_contig) {
            (true, true, _, _) => Variant::Dot,
            (true, _, true, _) => Variant::OuterNLhsContig,
            (_, _, true, true) => Variant::OuterNLhsStrided,
            (_, true, _, true) => Variant::OuterM,
            _ => unreachable!("layout combination not classifiable"),
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Variant {
    Dot,
    OuterNLhsContig,
    OuterNLhsStrided,
    OuterM,
}

impl Variant {
    /// Which output axis the planes within a cube enumerate. The planes
    /// step across this axis; the other axis is held constant within the
    /// cube and only advances via the cube grid.
    pub fn planes_split(self) -> PlanesSplit {
        match self {
            // Vector accumulator is along M — independent per N column,
            // so planes split N. (Same as Dot, just larger M-block.)
            Variant::OuterM => PlanesSplit::N,
            // Vector accumulator is along N — independent per M row, so
            // planes split M for OuterN. Dot has no constraint; pick N
            // so columnar work parallelizes naturally on wide outputs.
            Variant::OuterNLhsContig | Variant::OuterNLhsStrided => PlanesSplit::M,
            Variant::Dot => PlanesSplit::N,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlanesSplit {
    M,
    N,
}

fn operand_kind(dim: usize, layout: MatrixLayout) -> OperandLayout {
    if dim == 1 {
        OperandLayout::Vector
    } else {
        match layout {
            MatrixLayout::RowMajor => OperandLayout::RowMajor,
            MatrixLayout::ColMajor => OperandLayout::ColMajor,
        }
    }
}

/// Unified config for the gemm family. `kind` selects the kernel variant;
/// `plane_dim` is the hardware plane width (only `Variant::Dot` supports
/// `plane_dim > 1` — outer-product variants enforce `plane_dim == 1`).
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GemmConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) kind: MatmulOperandLayouts,
    pub(crate) planes_split: PlanesSplit,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for GemmConfig {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: layout_for(self.kind.lhs, MatrixLayout::RowMajor),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: layout_for(self.kind.rhs, MatrixLayout::ColMajor),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }
}

pub(crate) fn layout_for(operand: OperandLayout, vec_default: MatrixLayout) -> MatrixLayout {
    match operand {
        OperandLayout::RowMajor => MatrixLayout::RowMajor,
        OperandLayout::ColMajor => MatrixLayout::ColMajor,
        OperandLayout::Vector => vec_default,
    }
}
