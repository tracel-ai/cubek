use crate::attention::{D, DV, Q, S};
use crate::matmul::{
    MatmulAxes, MatmulDestination, MatmulLayout, MatmulOperands, MatmulRows, MatmulWorkItem,
};
use cubecl::ir::ElemType;
use cubecl::tile::Axis;

/// A work item of attention's backward pass as its microkernels see it: the element the inputs
/// and the gradients are stored in, the one everything is summed in, which the logsumexp and the
/// dot products are stored in too, the query heads that share each key head, and the dims a row
/// of the query and of the output spans, which every owner holds whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionBackwardWorkItem {
    pub element: ElemType,
    pub sum: ElemType,
    /// Query heads per key head.
    pub group: usize,
    pub head_dim: usize,
    pub val_dim: usize,
}

impl AttentionBackwardWorkItem {
    /// The score, `query · keyᵀ`, as a shared multiplication microkernel is handed it, its rows
    /// along `rows`: the queries, `Q` by `S`, for a walk owned by queries, or the keys, `S` by `Q`,
    /// for one owned by keys. Both factors lie contiguous along `D`, the contraction.
    pub fn score(&self, rows: Axis) -> MatmulWorkItem {
        self.stored_product(rows, Self::columns_beside(rows), D)
    }

    /// The gradient against the probabilities, `out_gradient · valueᵀ`, its rows along `rows` as
    /// the [`score`](Self::score)'s are. Both factors lie contiguous along `DV`, the contraction.
    pub fn score_gradient(&self, rows: Axis) -> MatmulWorkItem {
        self.stored_product(rows, Self::columns_beside(rows), DV)
    }

    /// The query's gradient, `score_gradient · key`: rows `Q` by columns `D`, contracting `S`, out
    /// of the score's gradient its owner keeps at `kept`. The keys lie contiguous along `D`, the
    /// columns.
    pub fn query_gradient(&self, kept: ElemType) -> MatmulWorkItem {
        self.kept_product(Q, D, S, kept)
    }

    /// The keys' gradient, `score_gradientᵀ · query`: rows `S` by columns `D`, contracting `Q`, out
    /// of the score's gradient its owner keeps keys first at `kept`. The queries lie contiguous
    /// along `D`.
    pub fn key_gradient(&self, kept: ElemType) -> MatmulWorkItem {
        self.kept_product(S, D, Q, kept)
    }

    /// The values' gradient, `probabilitiesᵀ · out_gradient`: rows `S` by columns `DV`, contracting
    /// `Q`, out of the probabilities their owner keeps keys first at `kept`. The output's gradient
    /// lies contiguous along `DV`.
    pub fn value_gradient(&self, kept: ElemType) -> MatmulWorkItem {
        self.kept_product(S, DV, Q, kept)
    }

    /// The axis a score held with its rows along `rows` has for columns.
    fn columns_beside(rows: Axis) -> Axis {
        match rows == Q {
            true => S,
            false => Q,
        }
    }

    /// A product of two stored factors, both contiguous along the contraction, into a sum its
    /// owner keeps: rows `m` by columns `n`, contracting `k`.
    fn stored_product(&self, m: Axis, n: Axis, k: Axis) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: MatmulOperands {
                lhs: self.element,
                rhs: self.element,
                out: self.sum,
            },
            sum: self.sum,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Col,
                ..MatmulOperands::all(MatmulLayout::Row)
            },
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes { m, n, k },
        }
    }

    /// A product out of a score its owner keeps at `kept`, against a stored rhs contiguous along
    /// the columns, into a gradient stored at the element: a unit keeps its score in registers at
    /// the sum's element, a plane casts it to the operands' for its fragments to load.
    fn kept_product(&self, m: Axis, n: Axis, k: Axis, kept: ElemType) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: MatmulOperands {
                lhs: kept,
                rhs: self.element,
                out: self.element,
            },
            sum: self.sum,
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes { m, n, k },
        }
    }
}
