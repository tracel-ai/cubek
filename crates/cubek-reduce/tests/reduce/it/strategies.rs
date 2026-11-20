//! Each kernel a strategy pins, and the kernels the autotune and the rule pick, through
//! `Reduce::execute`: a rank-3 tensor reduced along each of its dims by every operation, more row
//! axes than a launch grid has, the atomic hypercube's sums, and each strategy over many short
//! rows, one long row and rows across the contiguous dim. A kernel that declines a problem on the
//! device is a skip.

use cubecl::zspace::Shape;
use cubek_reduce::eval::cpu_reference::contiguous_strides;
use cubek_reduce::kernels::{ReduceKernel, ReduceOpStrategy};

use crate::reduce::it::test_case::TestCase;

/// The atomic hypercube: a row split across cubes, each part added into the zeroed output.
const ATOMIC: ReduceOpStrategy = ReduceOpStrategy::Specific(ReduceKernel::HypercubeAtomic);
/// The autotune, which measures every kernel the device serves and runs the fastest.
const AUTOTUNE: ReduceOpStrategy = ReduceOpStrategy::Autotune;

/// `shape`, contiguous, reduced along `axis` by the kernel `strategy` sends it to.
fn case(strategy: ReduceOpStrategy, shape: &[usize], axis: usize) -> TestCase {
    TestCase::new::<f32>(
        Shape::from(shape.to_vec()),
        contiguous_strides(shape),
        Some(axis),
        strategy,
    )
}

/// A test for each `name: check`, each one call of `check` on `shape` reduced along `axis` by
/// the kernel `strategy` sends it to.
macro_rules! reduction_tests {
    (
        $strategy:expr,
        $shape:expr,
        $axis:expr,
        [$($name:ident: $check:ident $(($k:expr))?),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                crate::reduce::it::strategies::case($strategy, &$shape, $axis).$check($($k)?);
            }
        )*
    };
}

/// [`reduction_tests`] for every operation whose rows hold what it writes; a top-3 needs three
/// values a row.
macro_rules! every_operation {
    ($strategy:expr, $shape:expr, $axis:expr) => {
        every_operation!($strategy, $shape, $axis, rows_shorter_than_three);
        reduction_tests!($strategy, $shape, $axis, [
            top_3: test_topk(3),
            arg_top_3: test_argtopk(3),
        ]);
    };
    ($strategy:expr, $shape:expr, $axis:expr, rows_shorter_than_three) => {
        reduction_tests!($strategy, $shape, $axis, [
            sum: test_sum,
            prod: test_prod,
            mean: test_mean,
            max: test_max,
            min: test_min,
            max_abs: test_max_abs,
            any: test_any,
            all: test_all,
            arg_max: test_argmax,
            arg_min: test_argmin,
            top_1: test_topk(1),
            arg_top_1: test_argtopk(1),
        ]);
    };
}

/// The kernel `strategy` pins reduces a `[2, 5, 300]` tensor along each dim by every operation,
/// and a `[2, 3, 2, 2, 64]` one, with more row axes than a launch grid has, along two of them.
macro_rules! reduces_each_dim {
    ($strategy:expr) => {
        /// Rows 300 long, along the contiguous dim.
        mod along_the_contiguous_dim {
            every_operation!($strategy, [2, 5, 300], 2);
        }

        /// Rows 5 long, across the contiguous dim.
        mod across_the_middle_dim {
            every_operation!($strategy, [2, 5, 300], 1);
        }

        /// Rows 2 long, across the contiguous dim, too short for a top-3.
        mod across_the_outer_dim {
            every_operation!($strategy, [2, 5, 300], 0, rows_shorter_than_three);
        }

        /// More row axes than a grid has, the rows on both sides of the reduced axis.
        mod more_row_axes_around_the_reduced_one {
            reduction_tests!($strategy, [2, 3, 2, 2, 64], 2, [
                sum: test_sum,
                mean: test_mean,
                arg_max: test_argmax,
            ]);
        }

        /// More row axes than a grid has, all before the reduced axis.
        mod more_row_axes_before_the_reduced_one {
            reduction_tests!($strategy, [2, 3, 2, 2, 64], 4, [
                sum: test_sum,
                mean: test_mean,
                arg_max: test_argmax,
            ]);
        }
    };
}

/// The unit kernel: a unit reduces each row.
mod unit_direct {
    reduces_each_dim!(crate::reduce::it::test_case::UNIT);
}

/// The plane kernel: a plane reduces each row, its lanes meeting by shuffles.
mod plane_shuffle {
    reduces_each_dim!(crate::reduce::it::test_case::PLANE);
}

/// The cube kernel: a cube reduces each row, its planes meeting in shared memory.
mod cube_staged {
    reduces_each_dim!(crate::reduce::it::test_case::CUBE);
}

/// The atomic hypercube sums rows split into parts, each part added into the zeroed output. A sum
/// is the only operation it serves.
mod hypercube_atomic {
    reduction_tests!(crate::reduce::it::strategies::ATOMIC, [1, 1, 9000], 2, [sums_one_long_row: test_sum]);
    reduction_tests!(crate::reduce::it::strategies::ATOMIC, [2, 3, 5000], 2, [sums_rows_along_the_contiguous_dim: test_sum]);
    reduction_tests!(crate::reduce::it::strategies::ATOMIC, [3, 2100, 4], 1, [sums_rows_across_the_contiguous_dim: test_sum]);
    reduction_tests!(crate::reduce::it::strategies::ATOMIC, [4, 9000], 1, [sums_the_rows_of_a_matrix: test_sum]);
}

/// Every strategy reduces like the CPU wherever it sends a problem: many short rows, one long row,
/// and rows across the contiguous dim, at extents that are not powers of two.
macro_rules! strategy_reduces_each_problem {
    ($strategy:expr) => {
        /// `[96, 7, 300]` along its contiguous dim: many short rows.
        mod many_short_rows {
            reduction_tests!($strategy, [96, 7, 300], 2, [
                sum: test_sum,
                mean: test_mean,
                arg_max: test_argmax,
                top_3: test_topk(3),
            ]);
        }

        /// `[1, 1, 9000]` along its contiguous dim: one long row.
        mod one_long_row {
            reduction_tests!($strategy, [1, 1, 9000], 2, [
                sum: test_sum,
                mean: test_mean,
                arg_max: test_argmax,
                top_3: test_topk(3),
            ]);
        }

        /// `[3, 2100, 5]` along its middle dim: rows across the contiguous dim.
        mod rows_across_the_contiguous_dim {
            reduction_tests!($strategy, [3, 2100, 5], 1, [
                sum: test_sum,
                mean: test_mean,
                arg_max: test_argmax,
                top_3: test_topk(3),
            ]);
        }
    };
}

/// The kernel each strategy picks, or the one it pins.
mod strategy {
    /// The autotune's pick.
    mod autotune {
        strategy_reduces_each_problem!(crate::reduce::it::strategies::AUTOTUNE);
    }

    /// The rule's pick.
    mod tile_heuristic {
        strategy_reduces_each_problem!(crate::reduce::it::test_case::RULE);
    }

    /// The unit kernel, pinned.
    mod unit {
        strategy_reduces_each_problem!(crate::reduce::it::test_case::UNIT);
    }

    /// The plane kernel, pinned.
    mod plane_shuffle {
        strategy_reduces_each_problem!(crate::reduce::it::test_case::PLANE);
    }

    /// The cube kernel, pinned.
    mod cube_staged {
        strategy_reduces_each_problem!(crate::reduce::it::test_case::CUBE);
    }

    /// The last-arrival hypercube, pinned.
    mod hypercube_last_arrival {
        strategy_reduces_each_problem!(crate::reduce::it::test_case::HYPERCUBE);
    }

    /// The atomic hypercube, pinned: it serves sums alone and declines the rest.
    mod hypercube_atomic {
        strategy_reduces_each_problem!(crate::reduce::it::strategies::ATOMIC);
    }
}
