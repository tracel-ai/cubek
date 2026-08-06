use cubecl::{
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
    ///
    /// For the instructions whose output is a coordinate (`Arg*`) or a logical flag
    /// (`Any` / `All`), `output` already is that element type, so the write is counted
    /// once whatever the instruction produces.
    pub dtypes: ReduceDtypes,
}

impl ReduceCost {
    /// Calculates compute operations and compulsory memory traffic for the reduction.
    ///
    /// Computes operations as `(reduce_len - 1) * ops_per_step` per fold, and byte traffic
    /// for input reads and output writes.
    pub fn work(&self) -> Work {
        let outputs = self.reduce_count * self.outputs_per_fold();

        Work {
            compute_ops: self.reduce_count
                * self.reduce_len.saturating_sub(1)
                * self.ops_per_step(),
            bytes: self.reduce_len * self.reduce_count * self.dtypes.input.size()
                + outputs * self.dtypes.output.size(),
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

    /// Minimum operations required per reduction step.
    ///
    /// Counted from the instructions the unit routine emits for one element, that
    /// routine being the cheapest fold (the plane and cube routines add plane
    /// reductions and accumulator fusions on top).
    fn ops_per_step(&self) -> usize {
        match self.instruction {
            // A single arithmetic operation: an add, or a multiply for `Prod`.
            // `Mean` folds as a sum and only divides once at the end.
            ReduceOperationConfig::Sum
            | ReduceOperationConfig::Prod
            | ReduceOperationConfig::Mean => 1,
            // A comparison and the select that keeps the winner.
            ReduceOperationConfig::Max | ReduceOperationConfig::Min => 2,
            // The same, over `abs` of the element.
            ReduceOperationConfig::MaxAbs => 3,
            // The element is first normalized to a flag, itself a comparison and a
            // select, before the same comparison and select fold it in.
            ReduceOperationConfig::Any | ReduceOperationConfig::All => 4,
            // Comparing values, breaking the tie on the coordinates, then selecting
            // the winning flag, value and coordinate.
            ReduceOperationConfig::ArgMax | ReduceOperationConfig::ArgMin => 6,
            // A sorted insertion walks all k slots, each a comparison and the two
            // selects that shift the displaced value along.
            ReduceOperationConfig::TopK(k) => 3 * k,
            // The same walk, carrying the coordinates: the tie-break costs two more
            // comparisons a slot, and the displaced coordinate two more selects.
            ReduceOperationConfig::ArgTopK(k) => 8 * k,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind, StorageType, UIntKind};

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
    fn tracking_an_index_costs_six_ops_a_step() {
        let argmax = ReduceCost {
            instruction: ReduceOperationConfig::ArgMax,
            ..cost()
        };

        assert_eq!(argmax.work().compute_ops, 6 * cost().work().compute_ops);
    }

    #[test]
    fn comparing_costs_an_op_more_than_accumulating() {
        let max = ReduceCost {
            instruction: ReduceOperationConfig::Max,
            ..cost()
        };

        assert_eq!(max.work().compute_ops, 2 * cost().work().compute_ops);
    }

    #[test]
    fn counts_the_input_once_and_one_output_a_fold() {
        // 12 * 5 input elements + 12 output elements * 4 bytes
        assert_eq!(cost().work().bytes, (60 + 12) * 4);
    }

    #[test]
    fn a_top_k_insertion_costs_three_ops_a_slot() {
        let topk = ReduceCost {
            instruction: ReduceOperationConfig::TopK(3),
            ..cost()
        };

        assert_eq!(topk.work().compute_ops, 9 * cost().work().compute_ops);
    }

    #[test]
    fn tracking_the_coordinates_costs_eight_ops_a_slot() {
        let argtopk = ReduceCost {
            instruction: ReduceOperationConfig::ArgTopK(3),
            ..cost()
        };

        assert_eq!(argtopk.work().compute_ops, 24 * cost().work().compute_ops);
    }

    #[test]
    fn normalizing_a_flag_costs_two_ops_more_than_comparing() {
        let any = ReduceCost {
            instruction: ReduceOperationConfig::Any,
            ..cost()
        };

        assert_eq!(any.work().compute_ops, 4 * cost().work().compute_ops);
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
    fn counts_a_coordinate_output_like_any_other_value() {
        // `Arg*` writes one u32 coordinate a fold, which `dtypes.output` already carries.
        let argmax = ReduceCost {
            instruction: ReduceOperationConfig::ArgMax,
            dtypes: ReduceDtypes {
                output: StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                ..f32_dtypes()
            },
            ..cost()
        };

        assert_eq!(argmax.work().bytes, (60 + 12) * 4);
    }
}
