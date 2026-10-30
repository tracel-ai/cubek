use crate::{
    ScanKind,
    kernels::{plane_exclusive_prefix_scan, plane_inclusive_prefix_scan},
};
use cubecl::prelude::*;

#[cube]
pub trait ScanInstruction: Send + Sync + 'static + std::fmt::Debug + CubeType + Sized {
    fn identity<T: Numeric>() -> T;
    fn combine<T: Numeric>(a: T, b: T) -> T;
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N>;
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: ScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T;
}

#[derive(CubeType, Debug)]
pub struct Sum {}
#[cube]
impl ScanInstruction for Sum {
    fn identity<T: Numeric>() -> T {
        T::from_int(0)
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        a + b
    }
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N> {
        a + b
    }
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: ScanKind,
        #[comptime] _plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            ScanKind::Exclusive => plane_exclusive_sum(value),
            ScanKind::Inclusive => plane_inclusive_sum(value),
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Product {}
#[cube]
impl ScanInstruction for Product {
    fn identity<T: Numeric>() -> T {
        T::from_int(1)
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        a * b
    }
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N> {
        a * b
    }
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: ScanKind,
        #[comptime] _plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            ScanKind::Exclusive => plane_exclusive_prod(value),
            ScanKind::Inclusive => plane_inclusive_prod(value),
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Minimum {}
#[cube]
impl ScanInstruction for Minimum {
    fn identity<T: Numeric>() -> T {
        T::max_value()
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        select(a < b, a, b)
    }
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N> {
        select(a < b, a, b)
    }
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: ScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            ScanKind::Exclusive => plane_exclusive_prefix_scan::<T, Self>(value, plane_size_log),
            ScanKind::Inclusive => plane_inclusive_prefix_scan::<T, Self>(value, plane_size_log),
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Maximum {}
#[cube]
impl ScanInstruction for Maximum {
    fn identity<T: Numeric>() -> T {
        T::min_value()
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        select(a > b, a, b)
    }
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N> {
        select(a > b, a, b)
    }
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: ScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T {
        match scan_kind {
            ScanKind::Exclusive => plane_exclusive_prefix_scan::<T, Self>(value, plane_size_log),
            ScanKind::Inclusive => plane_inclusive_prefix_scan::<T, Self>(value, plane_size_log),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanInstructionKind {
    Sum,
    Product,
    Minimum,
    Maximum,
}

impl ScanInstructionKind {
    #[cfg(feature = "benchmarks")]
    pub const ALL: &[ScanInstructionKind] = &[
        ScanInstructionKind::Sum,
        ScanInstructionKind::Product,
        ScanInstructionKind::Minimum,
        ScanInstructionKind::Maximum,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ScanInstructionKind::Sum => "Sum",
            ScanInstructionKind::Product => "Product",
            ScanInstructionKind::Minimum => "Min",
            ScanInstructionKind::Maximum => "Max",
        }
    }
}
