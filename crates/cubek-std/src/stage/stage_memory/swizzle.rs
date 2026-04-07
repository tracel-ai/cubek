use cubecl::{prelude::*, std::Swizzle};

/// Swizzling mode of the shared memory. Default `None`.
/// Matches the base TMA functionality, alternative chunk sizes or more complex patterns don't really
/// apply to matmul.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
pub enum SwizzleMode {
    /// No swizzling
    #[default]
    None,
    /// Swizzle 16B chunks within 32B span
    /// Swizzle<1,4,3>
    B32,
    /// Swizzle 16B chunks within 64B span
    /// Swizzle<2,4,3>
    B64,
    /// Swizzle 16B chunks within 128B span
    /// Swizzle<3,4,3>
    B128,
}

impl SwizzleMode {
    pub fn atom_size(&self) -> usize {
        match self {
            SwizzleMode::None => usize::MAX,
            SwizzleMode::B32 | SwizzleMode::B64 | SwizzleMode::B128 => 16,
        }
    }

    pub fn span_size(&self) -> usize {
        match self {
            SwizzleMode::None => 1,
            SwizzleMode::B32 => 32,
            SwizzleMode::B64 => 64,
            SwizzleMode::B128 => 128,
        }
    }
}

#[cube]
pub fn as_swizzle_object(#[comptime] mode: SwizzleMode) -> Swizzle {
    match mode {
        SwizzleMode::None => Swizzle::none(),
        SwizzleMode::B32 => Swizzle::new(1u32, 4u32, 3),
        SwizzleMode::B64 => Swizzle::new(2u32, 4u32, 3),
        SwizzleMode::B128 => Swizzle::new(3u32, 4u32, 3),
    }
}

impl From<SwizzleMode> for TensorMapSwizzle {
    fn from(value: SwizzleMode) -> Self {
        match value {
            SwizzleMode::None => TensorMapSwizzle::None,
            SwizzleMode::B32 => TensorMapSwizzle::B32,
            SwizzleMode::B64 => TensorMapSwizzle::B64,
            SwizzleMode::B128 => TensorMapSwizzle::B128,
        }
    }
}
