use cubek_std::MatrixLayout;

use crate::{
    components::{
        batch::{BatchConfig, CheckBounds},
        global::memory::GlobalLayoutConfig,
    },
    definition::{MatmulProblem, MatmulSetupError},
};

/// Role and layout of a single operand for the plane-parallel kernel.
///
/// An operand is either a matrix (with a memory layout) or a vector, when
/// its non-k dimension is 1.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum OperandLayout {
    RowMajor,
    ColMajor,
    Vector,
}

/// Shape + layout of the matmul problem the plane-parallel kernel will run.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MatmulOperandLayouts {
    pub lhs: OperandLayout,
    pub rhs: OperandLayout,
}

/// Which output axis the planes within a cube enumerate. The planes step
/// across this axis; the other axis is held constant within the cube and
/// only advances via the cube grid.
///
/// Forced for the single-staged paths (planes must walk the staged-MN
/// axis so they land on different staged tiles); a free routing choice
/// for Direct and DoubleStaged.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlanesSplit {
    /// Each plane in a cube takes a different M-row (or M-tile, when lhs
    /// is staged); all planes in the cube share the same N position.
    M,
    /// Each plane in a cube takes a different N-column (or N-tile, when
    /// rhs is staged); all planes in the cube share the same M position.
    N,
}

/// How a single operand's K axis is read.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum KAccess {
    /// K is contiguous → one vector load along K per K-tile.
    Direct,
    /// K is non-contiguous → load a tile and re-read it in the right direction.
    Staged,
}

/// Per-side K access. The four `(lhs, rhs)` combinations recover the
/// historical traversal names: `(Direct, Direct)` is the plain GEMM path,
/// `(Direct, Staged)` stages rhs, `(Staged, Direct)` stages lhs, and
/// `(Staged, Staged)` stages both (each plane produces a
/// `vector_size × vector_size` block of output).
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct KTraversal {
    pub lhs: KAccess,
    pub rhs: KAccess,
}

impl KTraversal {
    /// True iff at least one side needs the staged-tile path.
    pub fn is_staged(self) -> bool {
        matches!(self.lhs, KAccess::Staged) || matches!(self.rhs, KAccess::Staged)
    }
}

impl MatmulOperandLayouts {
    /// Infer the kind from problem shape and operand layouts.
    pub(crate) fn from_problem(
        problem: &MatmulProblem,
    ) -> Result<MatmulOperandLayouts, MatmulSetupError> {
        let lhs = operand_kind(problem.m, problem.lhs_layout);
        let rhs = operand_kind(problem.n, problem.rhs_layout);
        Ok(MatmulOperandLayouts { lhs, rhs })
    }

    /// Per-side K access. Staging is needed exactly when the operand's K
    /// axis is the non-contiguous one in memory:
    ///   - `ColMajor` lhs (K-stride = M) → stage lhs.
    ///   - `RowMajor` rhs (K-stride = N) → stage rhs.
    ///   - Vectors and the other layouts have K contiguous → direct.
    pub fn k_traversal(self) -> KTraversal {
        use KAccess::*;
        use OperandLayout::*;
        KTraversal {
            lhs: match self.lhs {
                RowMajor | Vector => Direct,
                ColMajor => Staged,
            },
            rhs: match self.rhs {
                RowMajor => Staged,
                ColMajor | Vector => Direct,
            },
        }
    }
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GemmPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) kind: MatmulOperandLayouts,
    pub(crate) planes_split: PlanesSplit,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for GemmPlaneParallelConfig {
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
