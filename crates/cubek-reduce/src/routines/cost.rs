use cubecl::{
    ir::StorageType,
    throughput::{ThroughputKey, ThroughputMode},
    tune::Work,
};

use crate::{
    components::instructions::ReduceOperationConfig, launch::ReduceDtypes, routines::ReduceProblem,
};

/// Minimal representation of reduce cost dependencies, including reduction length, operation, and element types.
#[derive(Debug, Clone, Copy)]
pub struct ReduceCost {
    /// Number of elements along the reduction axis.
    pub reduce_len: usize,
    /// Number of reduction instances.
    pub reduce_count: usize,
    /// The reduction operation.
    pub instruction: ReduceOperationConfig,
    /// Element types of input, output, and accumulator.
    pub dtypes: ReduceDtypes,
    /// Element type for optional index output.
    pub indices: Option<StorageType>,
}

impl ReduceCost {
    /// Calculates compute operations and compulsory memory traffic for the reduction.
    ///
    /// Computes operations as `(reduce_len - 1) * ops_per_step` per fold, and byte traffic
    /// for input reads and output/index writes.
    pub fn work(&self) -> Work {
        let outputs = self.reduce_count * self.outputs_per_fold();
        let indices_bytes = match self.indices {
            Some(indices) => outputs * indices.size(),
            None => 0,
        };

        Work {
            compute_ops: self.reduce_count
                * self.reduce_len.saturating_sub(1)
                * self.ops_per_step(),
            bytes: self.reduce_len * self.reduce_count * self.dtypes.input.size()
                + outputs * self.dtypes.output.size()
                + indices_bytes,
        }
    }

    /// Generates a throughput key using direct ALU throughput for the accumulation element type.
    pub fn compute_key(&self) -> ThroughputKey {
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect {
                dtype: self.dtypes.accumulation.elem_type(),
            },
        }
    }

    /// Number of output values written per fold.
    fn outputs_per_fold(&self) -> usize {
        match self.instruction {
            ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => k,
            _ => 1,
        }
    }

    /// Estimated minimum operations required per reduction step.
    fn ops_per_step(&self) -> usize {
        match self.instruction {
            // Single binary operation (e.g. sum, product, min, max, logical).
            ReduceOperationConfig::Sum
            | ReduceOperationConfig::Prod
            | ReduceOperationConfig::Mean
            | ReduceOperationConfig::Max
            | ReduceOperationConfig::Min
            | ReduceOperationConfig::MaxAbs
            | ReduceOperationConfig::Any
            | ReduceOperationConfig::All => 1,
            // Operations tracking index/position require comparison and coordinate updates.
            ReduceOperationConfig::ArgMax
            | ReduceOperationConfig::ArgMin
            | ReduceOperationConfig::ArgTopK(_)
            | ReduceOperationConfig::TopK(_) => 2,
        }
    }
}

impl From<&ReduceProblem> for ReduceCost {
    fn from(problem: &ReduceProblem) -> Self {
        Self {
            reduce_len: problem.reduce_len,
            reduce_count: problem.reduce_count,
            instruction: problem.instruction,
            dtypes: problem.dtypes,
            indices: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind, UIntKind};

    fn f32_dtypes() -> ReduceDtypes {
        let f32 = StorageType::Scalar(ElemType::Float(FloatKind::F32));

        ReduceDtypes {
            input: f32,
            output: f32,
            accumulation: f32,
        }
    }

    fn cost() -> ReduceCost {
        ReduceCost {
            reduce_len: 5,
            reduce_count: 12,
            instruction: ReduceOperationConfig::Sum,
            dtypes: f32_dtypes(),
            indices: None,
        }
    }

    #[test]
    fn folds_an_axis_with_one_op_less_than_its_length() {
        // 12 folds of 5 elements (4 accumulations per fold).
        assert_eq!(cost().work().compute_ops, 48);
    }

    #[test]
    fn an_axis_of_one_costs_nothing_to_fold() {
        let degenerate = ReduceCost {
            reduce_len: 1,
            ..cost()
        };

        assert_eq!(degenerate.work().compute_ops, 0);
    }

    #[test]
    fn an_empty_axis_costs_nothing_to_fold() {
        let empty = ReduceCost {
            reduce_len: 0,
            ..cost()
        };

        assert_eq!(empty.work().compute_ops, 0);
    }

    #[test]
    fn tracking_an_index_costs_two_ops_a_step() {
        let argmax = ReduceCost {
            instruction: ReduceOperationConfig::ArgMax,
            ..cost()
        };

        assert_eq!(argmax.work().compute_ops, 2 * cost().work().compute_ops);
    }

    #[test]
    fn counts_the_input_once_and_one_output_a_fold() {
        // 12 * 5 input elements + 12 output elements * 4 bytes
        assert_eq!(cost().work().bytes, (60 + 12) * 4);
    }

    #[test]
    fn top_k_writes_k_values_a_fold() {
        let topk = ReduceCost {
            instruction: ReduceOperationConfig::TopK(3),
            ..cost()
        };

        assert_eq!(topk.work().bytes, (60 + 12 * 3) * 4);
    }

    #[test]
    fn counts_the_indices_output_when_there_is_one() {
        let argmax = ReduceCost {
            instruction: ReduceOperationConfig::ArgMax,
            indices: Some(StorageType::Scalar(ElemType::UInt(UIntKind::U32))),
            ..cost()
        };

        assert_eq!(argmax.work().bytes, cost().work().bytes + 12 * 4);
    }
}

