//! The tests of a microkernel over a kind of rows, one launch each.

/// A test for each `name: operation`, each one launch of `microkernel` on rows `len` values long
/// holding `rows`, read `width` values a line, laid `lines`.
macro_rules! reduce_tests {
    (
        $microkernel:expr,
        $lines:ident,
        $len:expr,
        $width:expr,
        $rows:ident,
        [$($name:ident: $operation:expr),* $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                use cubek_test_utils::ScopeChecker;
                    $crate::reduce::testing::ReduceMicrokernelScopeChecker::new($microkernel).check([$crate::reduce::testing::ReduceCase {
                    operation: $operation,
                    value_count: $len,
                    lane_count: $width,
                    lines: $crate::reduce::testing::LineDirection::$lines,
                    rows: $crate::reduce::testing::Rows::$rows,
                }]);
            }
        )*
    };
}

/// [`reduce_tests`] on plain rows, for every operation a reduce microkernel serves.
macro_rules! every_operation {
    ($microkernel:expr, $lines:ident, $len:expr, $width:expr) => {
        use $crate::reduce::ReduceOperation;
        reduce_tests!($microkernel, $lines, $len, $width, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
            max_abs: ReduceOperation::MaxAbs,
            any: ReduceOperation::Any,
            all: ReduceOperation::All,
            arg_max: ReduceOperation::ArgMax,
            arg_min: ReduceOperation::ArgMin,
            top_3: ReduceOperation::TopK { k: 3 },
            arg_top_3: ReduceOperation::ArgTopK { k: 3 },
        ]);
    };
}

/// [`reduce_tests`] on the rows whose values a reduction must rank or test with care, each kind
/// with the operations it matters to.
macro_rules! edge_rows {
    ($microkernel:expr, $lines:ident, $len:expr, $width:expr) => {
        /// The largest and the smallest value twice: the lower index wins.
        mod ties {
            use super::*;
            use $crate::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Ties, [
                arg_max: ReduceOperation::ArgMax,
                arg_min: ReduceOperation::ArgMin,
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// A NaN twice: it ranks ahead of every number, so an extremum keeps the first.
        mod nan {
            use super::*;
            use $crate::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Nan, [
                max: ReduceOperation::Max,
                min: ReduceOperation::Min,
                max_abs: ReduceOperation::MaxAbs,
                arg_max: ReduceOperation::ArgMax,
                arg_min: ReduceOperation::ArgMin,
                top_3: ReduceOperation::TopK { k: 3 },
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// Every value `-inf`: a real value, which outranks every empty slot.
        mod negative_infinity {
            use super::*;
            use $crate::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, NegativeInfinity, [
                max: ReduceOperation::Max,
                arg_max: ReduceOperation::ArgMax,
                top_3: ReduceOperation::TopK { k: 3 },
                arg_top_3: ReduceOperation::ArgTopK { k: 3 },
            ]);
        }

        /// Every value zero: neither any nor all of them is set.
        mod zeros {
            use super::*;
            use $crate::reduce::ReduceOperation;
            reduce_tests!($microkernel, $lines, $len, $width, Zeros, [
                any: ReduceOperation::Any,
                all: ReduceOperation::All,
            ]);
        }
    };
}

/// `$kernel` launched as `$launch` binds it, `$extra` its sizes and Comptime Arguments before the
/// microkernel.
macro_rules! launch_component {
    ($launch:expr, $kernel:ident $(::<$family:ident>)?, sizes: [$($size:expr),*], [$($extra:expr),*], $microkernel:expr, $elements:expr $(,)?) => {{
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and the
        // kernel indexes no buffer but through its tiles.
        unsafe {
            $kernel::launch_unchecked $(::<$family>)? (
                $launch.client,
                $launch.launcher.cube_count(),
                $launch.launcher.cube_dim(),
                $launch.rows.line_lane_count,
                $($size,)*
                $launch.input,
                $launch.rows.bound_out($launch.out),
                $launch.launcher.partitioning_arg(),
                $($extra,)*
                $microkernel,
                $elements,
            )
        }
    }};
}

/// The launch function of a microkernel that reduces a region of a launch's rows, its method `reduce`.
macro_rules! region_reduce_kernel {
    ($kernel:ident, $microkernel:ty) => {
        /// Every region's rows, reduced by its owner.
        ///
        /// Its operands' leaves:
        /// - `input`: the region's rows, laid along or across the reduced dimension.
        /// - `out`: the region's cells, one a row (`k` for a top-k).
        #[cube(launch_unchecked)]
        fn $kernel<
            In: Numeric,
            Acc: Numeric,
            Out: Numeric,
            N: Size,
            K: Size,
            F: $crate::reduce::ReduceCarrierFamily,
        >(
            input: &cubecl::tile::TileTensor<'_>,
            out: &mut cubecl::tile::TileTensor<'_>,
            partitioning: cubecl::tile::Partitioning,
            #[comptime] microkernel: $microkernel,
            #[define(In, Acc, Out)] _elements: [cubecl::ir::ElemType; 3],
        ) {
            let x = input.tile(&partitioning);
            let o = out.tile_mut(&partitioning);

            for region in &partitioning {
                microkernel
                    .runtime()
                    .reduce::<In, Acc, Out, Out, N, F::Carrier<Acc, N, K>>(
                        &region,
                        &x,
                        &o,
                        &ComptimeOption::new_None(),
                    );
            }
        }
    };
}

/// A test that a microkernel's estimate is its item's work.
macro_rules! estimates_its_items_work {
    ($microkernel:expr) => {
        #[test]
        fn its_estimate_is_its_items_work() {
            let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
            let planes = CubePlanes {
                count: 4,
                width: 32,
            };
            let mut estimate = $crate::reduce::empty_estimate();
            $microkernel.estimate(&item, &(), planes, &mut estimate);
            assert_eq!(estimate.work, Some(item.unit_work()));
        }
    };
}
