//! The problem a routine computes.

use cubecl::ir::ElemType;
use cubecl::tile::Space;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubek_microkernel::rms_norm::{B, RmsNormOperands, RmsNormWorkItem};
use cubek_microkernel::row_store::RowStore;
use cubek_std::microkernel::{CubeStaging, GlobalStream};
use cubek_std::routine::EstimatedProblem;

/// `out[b, r] = input[b, r] / sqrt(mean(input[b, r]²) + epsilon) · weight` over `batch` entries
/// of `rows` rows of `dim` elements, as the host describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormRoutineProblem {
    pub batch: usize,
    pub rows: usize,
    pub dim: usize,
    /// The element every operand is stored in.
    pub element: ElemType,
    /// The alignment of each operand's buffer in elements, over the axes this kind names.
    pub elem_alignments: RmsNormOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: RmsNormOperands<u64>,
    /// How the output's rows are stored.
    pub store: RowStore,
}

impl RmsNormRoutineProblem {
    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> RmsNormWorkItem {
        RmsNormWorkItem {
            element: self.element,
            dim: self.dim,
            store: self.store,
        }
    }

    /// Every axis of the problem at its extent, its rows and its dim as the store states them:
    /// what a routine partitions.
    pub fn space(&self) -> Space {
        let axes = [
            vec![(B, self.batch)],
            self.store.rows_axes(self.rows),
            self.store.width_axes(self.dim),
        ]
        .concat();
        Space::new(&axes)
    }

    /// How one cube walks `rows` ([`RmsNormCubeRows`]): every row read once and written once, the
    /// weight read once, in one stage held while it computes.
    pub fn cube_staging(&self, rows: RmsNormCubeRows) -> CubeStaging {
        let size = self.element.size();
        let stream = |elements: usize| GlobalStream {
            bytes: (elements * size) as u64,
            run_bytes: rows.run * size,
        };
        CubeStaging {
            operations: (4 * rows.rows * self.dim) as f64,
            reads: vec![stream(rows.rows * self.dim), stream(self.dim)],
            writes: vec![stream(rows.rows * self.dim)],
            stages: 1,
            stages_in_flight: 1,
            drain_atomics: 0,
        }
    }

    /// A problem of dense operands of `element`, their views starting where their buffers do, one
    /// batch entry holding every row: what a key describes, and what a routine's pure tests
    /// prepare against.
    pub fn dense(rows: usize, dim: usize, element: ElemType) -> Self {
        let row = ElemAlignment::dense_rows(dim);
        Self {
            batch: 1,
            rows,
            dim,
            element,
            elem_alignments: RmsNormOperands {
                input: row,
                weight: row,
                out: row,
            },
            start_bytes: RmsNormOperands {
                input: 0,
                weight: 0,
                out: 0,
            },
            store: RowStore::Plain,
        }
    }

    /// This problem, its rows stored as `store` says.
    pub fn stored(self, store: RowStore) -> Self {
        Self { store, ..self }
    }
}

/// The rows one cube owns whole, and the run of a row each of its planes touches contiguously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormCubeRows {
    pub rows: usize,
    /// Consecutive elements of a row one owner reads: the row, or a cube's plane's run of it.
    pub run: usize,
}

impl EstimatedProblem for RmsNormRoutineProblem {
    /// Each row read once and written once, the weight read once, and four operations an element:
    /// the square, its sum, the scale and the weight.
    fn demand(&self) -> Work {
        let elements = self.batch * self.rows * self.dim;
        Work {
            compute_ops: 4 * elements,
            bytes: (2 * elements + self.dim) * self.element.size(),
        }
    }
}
