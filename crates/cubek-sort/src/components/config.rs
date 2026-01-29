//! Configuration constants and types for radix sort.

// ============================================================================
// Algorithm Constants
// ============================================================================

/// Bits processed per radix pass.
pub const RADIX_BITS: usize = 8;

/// Number of buckets per pass (2^RADIX_BITS = 256).
pub const NUM_BUCKETS: usize = 1 << RADIX_BITS;

/// Mask for extracting a digit value.
pub const DIGIT_MASK: u32 = (NUM_BUCKETS - 1) as u32;

// ============================================================================
// Compile-Time Configuration
// ============================================================================

/// Compile-time configuration for sort kernel compilation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SortBlueprint {
    /// Whether sorting keys only or key-value pairs.
    pub key_value_mode: KeyValueMode,
    /// Key transformation for proper unsigned ordering.
    pub key_transform: KeyTransform,
}

/// Sorting mode: keys only or key-value pairs.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum KeyValueMode {
    #[default]
    KeysOnly,
    KeyValue,
}

/// Transformation applied to keys for correct unsigned ordering.
///
/// Radix sort requires unsigned integer keys. This specifies how to
/// transform the original key type into a sortable unsigned form.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum KeyTransform {
    /// No transformation (unsigned integers).
    #[default]
    None,
    /// Flip sign bit (signed integers).
    SignedInt,
    /// Conditional bit flip (floats).
    Float,
}

// ============================================================================
// Runtime Configuration
// ============================================================================

/// Runtime tuning parameters for sort operations.
#[derive(Clone, Debug)]
pub struct SortStrategy {
    /// Elements processed per thread.
    pub items_per_thread: u32,
    /// Threads per block.
    pub threads_per_block: u32,
}

impl Default for SortStrategy {
    fn default() -> Self {
        Self {
            items_per_thread: 4,
            threads_per_block: 256,
        }
    }
}

impl SortStrategy {
    /// Elements processed per block.
    pub fn items_per_block(&self) -> u32 {
        self.items_per_thread * self.threads_per_block
    }

    /// Number of blocks needed for the given item count.
    pub fn num_blocks(&self, num_items: usize) -> u32 {
        let items_per_block = self.items_per_block() as usize;
        num_items.div_ceil(items_per_block) as u32
    }
}
