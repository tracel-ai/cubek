use cubecl::{
    client::ComputeClient,
    prelude::Runtime,
    throughput::{ThroughputKey, compute_throughput_key, select_cmma_tile},
    tune::Work,
};

use crate::definition::{MatmulElems, MatmulGlobalElems, MatmulProblem};

/// Minimal representation of matmul cost dependencies, including dimensions and element types.
#[derive(Debug, Clone)]
pub struct MatmulCost {
    /// Number of independent matmuls in the batch.
    pub batches: usize,
    /// Number of output rows (M).
    pub m: usize,
    /// Number of output columns (N).
    pub n: usize,
    /// Contracted dimension (K).
    pub k: usize,
    /// Global element types of the operands.
    pub elems: MatmulGlobalElems,
}

impl MatmulCost {
    /// Calculates the compute operations and compulsory memory traffic for the matmul.
    ///
    /// Computes operations as `2 * k - 1` ops per output element and compulsory byte traffic
    /// for reading inputs and writing outputs once.
    pub fn work(&self) -> Work {
        let elements = |rows: usize, cols: usize| self.batches * rows * cols;

        Work {
            compute_ops: elements(self.m, self.n) * (2 * self.k).saturating_sub(1),
            bytes: elements(self.m, self.k) * self.elems.lhs.size()
                + elements(self.k, self.n) * self.elems.rhs.size()
                + elements(self.m, self.n) * self.elems.out.size(),
        }
    }

    /// Generates a throughput key for compute probes representing peak hardware instruction throughput.
    ///
    /// Selects CMMA tiles without problem-size constraints to ensure small dimensions (e.g. `m = 1`)
    /// evaluate against peak hardware throughput, using register element types.
    pub fn compute_key<R: Runtime>(&self, client: &ComputeClient<R>) -> ThroughputKey {
        // Unconstrained selection allows matching hardware capability without problem-size padding limits.
        const UNCONSTRAINED: (usize, usize, usize) = (usize::MAX, usize::MAX, usize::MAX);

        let elems = MatmulElems::from_globals(&self.elems);

        let cmma_tile = select_cmma_tile(
            client,
            elems.lhs_register,
            elems.rhs_register,
            elems.acc_register,
            UNCONSTRAINED,
        );

        compute_throughput_key(
            cmma_tile,
            elems.lhs_register.elem_type(),
            elems.acc_register.elem_type(),
        )
    }
}

impl From<&MatmulProblem> for MatmulCost {
    fn from(problem: &MatmulProblem) -> Self {
        Self {
            batches: problem.out_batches.iter().product(),
            m: problem.m,
            n: problem.n,
            k: problem.k,
            elems: problem.global_dtypes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind, StorageType};

    fn f32_elems() -> MatmulGlobalElems {
        let f32 = StorageType::Scalar(ElemType::Float(FloatKind::F32));

        MatmulGlobalElems {
            lhs: f32,
            rhs: f32,
            out: f32,
        }
    }

    fn cost() -> MatmulCost {
        MatmulCost {
            batches: 1,
            m: 2,
            n: 3,
            k: 4,
            elems: f32_elems(),
        }
    }

    #[test]
    fn counts_a_multiply_and_an_add_per_contracted_element() {
        // 2x3 outputs with length-4 dot product: 4 multiplies and 3 adds per element.
        assert_eq!(cost().work().compute_ops, 6 * 7);
    }

    #[test]
    fn scales_the_op_count_with_the_batch() {
        let batched = MatmulCost {
            batches: 8,
            ..cost()
        };

        assert_eq!(batched.work().compute_ops, 8 * cost().work().compute_ops);
    }

    #[test]
    fn a_contraction_of_one_costs_one_op_per_output() {
        // k = 1 evaluates to 1 operation per output element.
        let dot = MatmulCost { k: 1, ..cost() };

        assert_eq!(dot.work().compute_ops, 6);
    }

    #[test]
    fn counts_every_operand_once_in_the_byte_total() {
        // (m*k + k*n + m*n) elements * 4 bytes
        assert_eq!(cost().work().bytes, 26 * 4);
    }

    #[test]
    fn counts_the_batch_in_every_operand() {
        let batched = MatmulCost {
            batches: 8,
            ..cost()
        };

        assert_eq!(batched.work().bytes, 8 * cost().work().bytes);
    }
}
