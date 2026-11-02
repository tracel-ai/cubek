//! The problem a routine computes.

use cubecl::ir::ElemType;
use cubecl::tile::Space;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubek_microkernel::row_store::RowStore;
use cubek_microkernel::swiglu::{H, R, SwigluOperands, SwigluWorkItem};
use cubek_std::microkernel::{CubeStaging, GlobalStream};
use cubek_std::routine::EstimatedProblem;

/// `out[r] = silu(gate[r]) ⊙ up[r]` over `rows` rows of `dim` elements, the gate and the up the
/// two halves of each fused row, as the host describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwigluRoutineProblem {
    pub rows: usize,
    /// The width of a half, which the output takes.
    pub dim: usize,
    /// The element every operand is stored in.
    pub element: ElemType,
    /// The alignment of each operand's buffer in elements, over the axes this kind names.
    pub elem_alignments: SwigluOperands<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: SwigluOperands<u64>,
    /// How the output's rows are stored.
    pub store: RowStore,
}

impl SwigluRoutineProblem {
    /// Whether the problem has no element to gate: no rows, or rows of no width.
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.dim == 0
    }

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> SwigluWorkItem {
        SwigluWorkItem {
            element: self.element,
            dim: self.dim,
            store: self.store,
        }
    }

    /// Every axis of the problem at its extent, its rows and its width as the store states them:
    /// what a routine partitions.
    pub fn space(&self) -> Space {
        let axes = [
            self.store.rows_axes(self.rows),
            vec![(H, 2)],
            self.store.width_axes(self.dim),
        ]
        .concat();
        Space::new(&axes)
    }

    /// How one cube walks its box, `cube_box`, rows by a run of their width: each line of either
    /// half read once and of the output written once, in one stage held while it computes.
    pub fn cube_staging(&self, cube_box: &Space) -> CubeStaging {
        let rows = cube_box.extent(R);
        let run = cube_box.cells() / rows / 2;
        let size = self.element.size();
        let stream = || GlobalStream {
            bytes: (rows * run * size) as u64,
            run_bytes: run * size,
        };
        CubeStaging {
            operations: (Self::OPERATIONS * rows * run) as f64,
            reads: vec![stream(), stream()],
            writes: vec![stream()],
            stages: 1,
            stages_in_flight: 1,
            drain_atomics: 0,
        }
    }

    /// A problem of fused rows of `2 · dim` dense elements of `element`, starting where their
    /// buffer does, and an output of dense rows: what a key describes, and what a routine's pure
    /// tests prepare against.
    pub fn dense(rows: usize, dim: usize, element: ElemType) -> Self {
        Self {
            rows,
            dim,
            element,
            elem_alignments: SwigluOperands {
                input: ElemAlignment::new(&[rows, 2, dim], &[2 * dim, dim, 1]),
                out: ElemAlignment::dense_rows(dim),
            },
            start_bytes: SwigluOperands { input: 0, out: 0 },
            store: RowStore::Plain,
        }
    }

    /// This problem, its output's rows stored as `store` says.
    pub fn stored(self, store: RowStore) -> Self {
        Self { store, ..self }
    }

    /// The operations an element costs ([`demand`](EstimatedProblem::demand) names them).
    const OPERATIONS: usize = 5;
}

impl EstimatedProblem for SwigluRoutineProblem {
    /// Each half read once and the output written once, at five operations an element: the
    /// negation, the exponential, the sum, the division and the product.
    fn demand(&self) -> Work {
        let elements = self.rows * self.dim;
        Work {
            compute_ops: Self::OPERATIONS * elements,
            bytes: 3 * elements * self.element.size(),
        }
    }
}
