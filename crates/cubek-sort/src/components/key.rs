use cubecl::prelude::*;

/// Trait for types that can be used as sort keys in CubeCL kernels.
#[cube]
pub trait SortKey: CubePrimitive {
    fn to_radix(value: Self) -> u32;
    fn from_radix(value: u32) -> Self;
}

#[cube]
impl SortKey for u32 {
    fn to_radix(value: u32) -> u32 {
        value
    }

    fn from_radix(value: u32) -> u32 {
        value
    }
}

#[cube]
impl SortKey for i32 {
    fn to_radix(value: i32) -> u32 {
        (value as u32) ^ 0x8000_0000u32
    }

    fn from_radix(value: u32) -> i32 {
        (value ^ 0x8000_0000u32) as i32
    }
}

#[cube]
impl SortKey for f32 {
    fn to_radix(value: f32) -> u32 {
        let bits = f32::to_bits(value);
        let mask = ((bits as i32) >> 31) as u32 | 0x8000_0000u32;
        bits ^ mask
    }

    fn from_radix(value: u32) -> f32 {
        let mask = (((value >> 31) as i32) - 1) as u32 | 0x8000_0000u32;
        f32::from_bits(value ^ mask)
    }
}

/// Number of radix sort passes for a key type (one pass per byte).
pub const fn num_passes<K>() -> u32 {
    core::mem::size_of::<K>() as u32
}
