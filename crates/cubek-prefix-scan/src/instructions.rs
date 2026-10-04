use crate::{
    definitions::PrefixScanKind,
    kernels::{plane_exclusive_prefix_scan, plane_inclusive_prefix_scan},
};
use cubecl::prelude::*;

#[cube]
pub trait PrefixScanInstruction:
    Send + Sync + 'static + std::fmt::Debug + CubeType + Sized
{
    fn identity<T: Numeric>() -> T;
    fn combine<T: Numeric>(a: T, b: T) -> T;
    fn combine_vector<T: Numeric, N: Size>(a: Vector<T, N>, b: Vector<T, N>) -> Vector<T, N>;
    fn plane_prefix_scan<T: Numeric>(
        value: T,
        #[comptime] scan_kind: PrefixScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T;
}

#[derive(CubeType, Debug)]
pub struct Sum {}
#[cube]
impl PrefixScanInstruction for Sum {
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
        #[comptime] scan_kind: PrefixScanKind,
        #[comptime] _plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            PrefixScanKind::Exclusive => plane_exclusive_sum(value),
            PrefixScanKind::Inclusive => plane_inclusive_sum(value),
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Product {}
#[cube]
impl PrefixScanInstruction for Product {
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
        #[comptime] scan_kind: PrefixScanKind,
        #[comptime] _plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            PrefixScanKind::Exclusive => plane_exclusive_prod(value),
            PrefixScanKind::Inclusive => plane_inclusive_prod(value),
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Minimum {}
#[cube]
impl PrefixScanInstruction for Minimum {
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
        #[comptime] scan_kind: PrefixScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T {
        #[comptime]
        match scan_kind {
            PrefixScanKind::Exclusive => {
                plane_exclusive_prefix_scan::<T, Self>(value, plane_size_log)
            }
            PrefixScanKind::Inclusive => {
                plane_inclusive_prefix_scan::<T, Self>(value, plane_size_log)
            }
        }
    }
}

#[derive(CubeType, Debug)]
pub struct Maximum {}
#[cube]
impl PrefixScanInstruction for Maximum {
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
        #[comptime] scan_kind: PrefixScanKind,
        #[comptime] plane_size_log: u32,
    ) -> T {
        match scan_kind {
            PrefixScanKind::Exclusive => {
                plane_exclusive_prefix_scan::<T, Self>(value, plane_size_log)
            }
            PrefixScanKind::Inclusive => {
                plane_inclusive_prefix_scan::<T, Self>(value, plane_size_log)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefixPrefixScanInstructionKind {
    Sum,
    Product,
    Minimum,
    Maximum,
}

impl PrefixPrefixScanInstructionKind {
    #[cfg(feature = "benchmarks")]
    pub const ALL: &[PrefixPrefixScanInstructionKind] = &[
        PrefixPrefixScanInstructionKind::Sum,
        PrefixPrefixScanInstructionKind::Product,
        PrefixPrefixScanInstructionKind::Minimum,
        PrefixPrefixScanInstructionKind::Maximum,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PrefixPrefixScanInstructionKind::Sum => "Sum",
            PrefixPrefixScanInstructionKind::Product => "Product",
            PrefixPrefixScanInstructionKind::Minimum => "Min",
            PrefixPrefixScanInstructionKind::Maximum => "Max",
        }
    }
}
