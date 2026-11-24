//! Top-k with a row count that is not a multiple of the output's vector width.
//!
//! The rows lie side by side across the reduced dim, so a line of the output is a line of rows and
//! each k slot is one contiguous run of them. With 9 rows and `k = 4` the whole output holds 36
//! scalars, a multiple of 4, while a run of 9 is not: a vector of 4 rows starting at row 8 would
//! spill into the next slot's run.

use cubecl::zspace::{Shape, Strides};
use cubek_reduce::kernels::ReduceOpStrategy;
use cubek_test_utils::ScopeChecker;

use super::test_case::{CUBE, Op, PLANE, RULE, ReduceScopeChecker, TestCase, UNIT};

/// `rows` rows 64 values long, side by side, reduced by `strategy` across their contiguous dim.
fn case(strategy: ReduceOpStrategy, rows: usize) -> TestCase {
    TestCase::new::<f32>(
        Shape::new([64, rows]),
        Strides::new(&[rows, 1]),
        Some(0),
        strategy,
    )
}

/// A test for each top-`k` of `rows` rows through each of the unit kernel, the rule, and the
/// plane and cube kernels, which skip where the device has no planes; each one launch: its values,
/// its indices, and both at once.
macro_rules! top_k_tests {
    ($rows:expr, $k:expr) => {
        top_k_tests!(@strategy unit, UNIT, $rows, $k);
        top_k_tests!(@strategy rule, RULE, $rows, $k);
        top_k_tests!(@strategy plane, PLANE, $rows, $k);
        top_k_tests!(@strategy cube, CUBE, $rows, $k);
    };
    (@strategy $name:ident, $strategy:ident, $rows:expr, $k:expr) => {
        mod $name {
            use super::super::*;

            #[test]
            fn top_k() {
                ReduceScopeChecker.check([case($strategy, $rows).reduced_by(Op::TopK($k))]);
            }

            #[test]
            fn arg_top_k() {
                ReduceScopeChecker.check([case($strategy, $rows).reduced_by(Op::ArgTopK($k))]);
            }

            #[test]
            fn top_k_with_indices() {
                ReduceScopeChecker.check([case($strategy, $rows).reduced_by(Op::TopKWithIndices($k))]);
            }
        }
    };
}

mod rows_not_multiple_of_vector_k4 {
    top_k_tests!(9, 4);
}

mod rows_not_multiple_of_vector_k2 {
    top_k_tests!(6, 2);
}

mod rows_multiple_of_vector_k4 {
    top_k_tests!(8, 4);
}
