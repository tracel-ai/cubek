/// Number of bits processed per radix sort pass.
pub const RADIX_BITS: usize = 8;

/// Number of buckets (2^RADIX_BITS).
pub const NUM_BUCKETS: usize = 1 << RADIX_BITS;

/// Mask for extracting a digit.
pub const DIGIT_MASK: u32 = (NUM_BUCKETS - 1) as u32;

/// Compile-time configuration for sort kernels.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SortBlueprint {
    /// Whether sorting keys only or key-value pairs.
    pub key_value_mode: KeyValueMode,
    /// The type of key transformation required.
    pub key_transform: KeyTransform,
}

/// Whether sorting keys only or key-value pairs.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum KeyValueMode {
    /// Sort keys only.
    #[default]
    KeysOnly,
    /// Sort key-value pairs together.
    KeyValue,
}

/// The transformation applied to keys before sorting.
///
/// Radix sort requires unsigned integer keys. This enum specifies
/// how to transform the original key type into a sortable unsigned form.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum KeyTransform {
    /// No transformation needed (unsigned integers).
    #[default]
    None,
    /// Flip the sign bit (signed integers).
    SignedInt,
    /// Conditional bit flip for proper float ordering.
    Float,
}

/// Runtime configuration for sort operations.
#[derive(Clone, Debug)]
pub struct SortStrategy {
    /// Number of elements each thread processes.
    pub items_per_thread: u32,
    /// Number of threads per thread block.
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
    /// Elements processed per thread block.
    pub fn items_per_block(&self) -> u32 {
        self.items_per_thread * self.threads_per_block
    }

    /// Calculate the number of thread blocks needed for a given number of items.
    pub fn num_blocks(&self, num_items: usize) -> u32 {
        let items_per_block = self.items_per_block() as usize;
        num_items.div_ceil(items_per_block) as u32
    }
}
