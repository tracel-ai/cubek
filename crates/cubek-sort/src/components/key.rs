use super::config::KeyTransform;
use cubecl::prelude::*;

/// Trait for types that can be used as sort keys.
///
/// This trait defines how to convert a key type to and from an unsigned
/// integer representation suitable for radix sorting.
pub trait SortKey: Numeric + CubeType + Send + Sync + 'static {
    /// The transformation required for this key type.
    const TRANSFORM: KeyTransform;

    /// Number of radix passes needed (4 for 32-bit, 2 for 16-bit).
    const NUM_PASSES: u32;

    /// The unsigned integer type used for radix sorting.
    type Unsigned: Numeric + CubeType;

    /// Convert a key to its radix-sortable unsigned representation.
    fn to_radix(value: Self) -> Self::Unsigned;

    /// Convert back from radix representation to the original key type.
    fn from_radix(value: Self::Unsigned) -> Self;
}

// Implementation for u32 - identity transform
impl SortKey for u32 {
    const TRANSFORM: KeyTransform = KeyTransform::None;
    const NUM_PASSES: u32 = 4;
    type Unsigned = u32;

    #[inline]
    fn to_radix(value: Self) -> Self::Unsigned {
        value
    }

    #[inline]
    fn from_radix(value: Self::Unsigned) -> Self {
        value
    }
}

// Implementation for i32 - flip sign bit
impl SortKey for i32 {
    const TRANSFORM: KeyTransform = KeyTransform::SignedInt;
    const NUM_PASSES: u32 = 4;
    type Unsigned = u32;

    #[inline]
    fn to_radix(value: Self) -> Self::Unsigned {
        // Flip the sign bit to convert signed range to unsigned range
        // This maps: i32::MIN -> 0, -1 -> 0x7FFFFFFF, 0 -> 0x80000000, i32::MAX -> 0xFFFFFFFF
        (value as u32) ^ 0x8000_0000
    }

    #[inline]
    fn from_radix(value: Self::Unsigned) -> Self {
        // Reverse the transformation
        (value ^ 0x8000_0000) as i32
    }
}

// Implementation for f32 - conditional bit flip
impl SortKey for f32 {
    const TRANSFORM: KeyTransform = KeyTransform::Float;
    const NUM_PASSES: u32 = 4;
    type Unsigned = u32;

    #[inline]
    fn to_radix(value: Self) -> Self::Unsigned {
        let bits = value.to_bits();
        // If sign bit is set (negative), flip all bits
        // If sign bit is clear (positive), flip only sign bit
        // This ensures proper ordering: -inf < -1 < -0 < +0 < +1 < +inf
        let mask = ((bits as i32) >> 31) as u32 | 0x8000_0000;
        bits ^ mask
    }

    #[inline]
    fn from_radix(value: Self::Unsigned) -> Self {
        // Reverse the transformation
        let mask = (((value >> 31) as i32) - 1) as u32 | 0x8000_0000;
        f32::from_bits(value ^ mask)
    }
}

/// CubeCL kernel function to extract a digit from a radix key.
#[cube]
pub fn extract_digit(key: u32, pass: u32) -> u32 {
    (key >> (pass * 8)) & 0xFFu32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_roundtrip() {
        let values = [0u32, 1, 100, u32::MAX / 2, u32::MAX];
        for v in values {
            assert_eq!(u32::from_radix(u32::to_radix(v)), v);
        }
    }

    #[test]
    fn test_i32_roundtrip() {
        let values = [i32::MIN, -100, -1, 0, 1, 100, i32::MAX];
        for v in values {
            assert_eq!(i32::from_radix(i32::to_radix(v)), v);
        }
    }

    #[test]
    fn test_i32_ordering() {
        // Verify that the radix representation preserves ordering
        let values = [i32::MIN, -100, -1, 0, 1, 100, i32::MAX];
        let radix: Vec<u32> = values.iter().map(|&v| i32::to_radix(v)).collect();
        for i in 0..radix.len() - 1 {
            assert!(
                radix[i] < radix[i + 1],
                "Ordering failed: {} -> {} vs {} -> {}",
                values[i],
                radix[i],
                values[i + 1],
                radix[i + 1]
            );
        }
    }

    #[test]
    fn test_f32_roundtrip() {
        let values = [
            f32::NEG_INFINITY,
            -1000.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1000.0,
            f32::INFINITY,
        ];
        for v in values {
            let result = f32::from_radix(f32::to_radix(v));
            assert!(
                (result == v) || (result.is_nan() && v.is_nan()),
                "Roundtrip failed for {}: got {}",
                v,
                result
            );
        }
    }

    #[test]
    fn test_f32_ordering() {
        // Verify that the radix representation preserves ordering
        let values = [
            f32::NEG_INFINITY,
            -1000.0,
            -1.0,
            -f32::MIN_POSITIVE,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            1000.0,
            f32::INFINITY,
        ];
        let radix: Vec<u32> = values.iter().map(|&v| f32::to_radix(v)).collect();
        for i in 0..radix.len() - 1 {
            assert!(
                radix[i] < radix[i + 1],
                "Ordering failed: {} -> {} vs {} -> {}",
                values[i],
                radix[i],
                values[i + 1],
                radix[i + 1]
            );
        }
    }
}
