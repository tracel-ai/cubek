use cubecl::prelude::*;
use half::{bf16, f16};

/// Trait for unsigned integer types that can be used as radix representations.
pub trait Radix: Int + Numeric + CubePrimitive + Sized + Send + Sync + 'static {}

impl Radix for u8 {}
impl Radix for u16 {}
impl Radix for u32 {}
impl Radix for u64 {}

/// Trait for types that can be used as sort keys in CubeCL kernels.
///
/// Each key type has an associated `Radix` type that determines the unsigned integer
/// representation used for sorting. The sort uses one pass per byte of the radix type.
#[cube]
pub trait SortKey: CubePrimitive {
    type Radix: Radix;
    fn to_radix(value: Self) -> Self::Radix;
    fn from_radix(value: Self::Radix) -> Self;
}

// 32-bit unsigned integer
#[cube]
impl SortKey for u32 {
    type Radix = u32;

    fn to_radix(value: u32) -> u32 {
        value
    }

    fn from_radix(value: u32) -> u32 {
        value
    }
}

// 32-bit signed integer
#[cube]
impl SortKey for i32 {
    type Radix = u32;

    fn to_radix(value: i32) -> u32 {
        (value as u32) ^ 0x8000_0000u32
    }

    fn from_radix(value: u32) -> i32 {
        (value ^ 0x8000_0000u32) as i32
    }
}

// 32-bit float (IEEE 754 single precision)
#[cube]
impl SortKey for f32 {
    type Radix = u32;

    fn to_radix(value: f32) -> u32 {
        let bits = f32::to_bits(value);
        // For positive floats: flip sign bit to make them sort after negatives
        // For negative floats: flip all bits to reverse their order
        let mask = ((bits as i32) >> 31) as u32 | 0x8000_0000u32;
        bits ^ mask
    }

    fn from_radix(value: u32) -> f32 {
        let mask = (((value >> 31) as i32) - 1) as u32 | 0x8000_0000u32;
        f32::from_bits(value ^ mask)
    }
}

// 8-bit unsigned integer
#[cube]
impl SortKey for u8 {
    type Radix = u8;

    fn to_radix(value: u8) -> u8 {
        value
    }

    fn from_radix(value: u8) -> u8 {
        value
    }
}

// 8-bit signed integer
#[cube]
impl SortKey for i8 {
    type Radix = u8;

    fn to_radix(value: i8) -> u8 {
        // Flip sign bit to make signed order match unsigned order
        (value as u8) ^ 0x80u8
    }

    fn from_radix(value: u8) -> i8 {
        (value ^ 0x80u8) as i8
    }
}

// 16-bit unsigned integer
#[cube]
impl SortKey for u16 {
    type Radix = u16;

    fn to_radix(value: u16) -> u16 {
        value
    }

    fn from_radix(value: u16) -> u16 {
        value
    }
}

// 16-bit signed integer
#[cube]
impl SortKey for i16 {
    type Radix = u16;

    fn to_radix(value: i16) -> u16 {
        // Flip sign bit to make signed order match unsigned order
        (value as u16) ^ 0x8000u16
    }

    fn from_radix(value: u16) -> i16 {
        (value ^ 0x8000u16) as i16
    }
}

// 16-bit float (IEEE 754 half precision)
#[cube]
impl SortKey for f16 {
    type Radix = u16;

    fn to_radix(value: f16) -> u16 {
        let bits = f16::to_bits(value);
        // Same sign-magnitude to unsigned conversion as f32, but for 16 bits
        let mask = ((bits as i16) >> 15) as u16 | 0x8000u16;
        bits ^ mask
    }

    fn from_radix(value: u16) -> f16 {
        let mask = (((value >> 15) as i16) - 1) as u16 | 0x8000u16;
        f16::from_bits(value ^ mask)
    }
}

// 16-bit bfloat
#[cube]
impl SortKey for bf16 {
    type Radix = u16;

    fn to_radix(value: bf16) -> u16 {
        let bits = bf16::to_bits(value);
        // Same sign-magnitude to unsigned conversion as f32, but for 16 bits
        let mask = ((bits as i16) >> 15) as u16 | 0x8000u16;
        bits ^ mask
    }

    fn from_radix(value: u16) -> bf16 {
        let mask = (((value >> 15) as i16) - 1) as u16 | 0x8000u16;
        bf16::from_bits(value ^ mask)
    }
}

#[cube]
impl SortKey for u64 {
    type Radix = u64;

    fn to_radix(value: u64) -> u64 {
        value
    }

    fn from_radix(value: u64) -> u64 {
        value
    }
}

#[cube]
impl SortKey for i64 {
    type Radix = u64;

    fn to_radix(value: i64) -> u64 {
        // Flip sign bit to make signed order match unsigned order
        (value as u64) ^ 0x8000_0000_0000_0000u64
    }

    fn from_radix(value: u64) -> i64 {
        (value ^ 0x8000_0000_0000_0000u64) as i64
    }
}

#[cube]
impl SortKey for f64 {
    type Radix = u64;

    fn to_radix(value: f64) -> u64 {
        let bits = f64::to_bits(value);
        // For positive floats: flip sign bit to make them sort after negatives
        // For negative floats: flip all bits to reverse their order
        let mask = ((bits as i64) >> 63) as u64 | 0x8000_0000_0000_0000u64;
        bits ^ mask
    }

    fn from_radix(value: u64) -> f64 {
        let mask = (((value >> 63) as i64) - 1) as u64 | 0x8000_0000_0000_0000u64;
        f64::from_bits(value ^ mask)
    }
}
