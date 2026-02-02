pub const RADIX_BITS: usize = 8;
pub const NUM_BUCKETS: usize = 1 << RADIX_BITS;

#[derive(Clone, Debug)]
pub struct SortStrategy {
    pub items_per_thread: u32,
    pub threads_per_block: u32,
}

impl Default for SortStrategy {
    fn default() -> Self {
        // 512 threads × 16 items = 8192 items per block
        // Larger blocks reduce scan overhead at the cost of occupancy
        Self {
            items_per_thread: 16,
            threads_per_block: 512,
        }
    }
}

// The current SortStrategy is a very simple heuristic. Instead we can probably use autotuning
// in Burn to figure out settings. Eg. we could choose a low, mid and high blocksize that we tune.
// Alternatively we figure out a more principled way to set these.
impl SortStrategy {
    /// Create a strategy optimized for keys-only sorting at the given input size.
    pub fn for_keys(num_items: usize) -> Self {
        if num_items < 4_000_000 {
            // Small inputs: 256 threads × 15 items = 3840 items/block
            // Smaller blocks improve GPU occupancy
            Self {
                items_per_thread: 15,
                threads_per_block: 256,
            }
        } else {
            // Large inputs: use default (8192 items/block)
            Self::default()
        }
    }

    /// Create a strategy optimized for key-value pair sorting at the given input size.
    pub fn for_pairs(_num_items: usize) -> Self {
        Self::default()
    }

    pub fn items_per_block(&self) -> u32 {
        self.items_per_thread * self.threads_per_block
    }

    pub fn num_blocks(&self, num_items: usize) -> u32 {
        num_items.div_ceil(self.items_per_block() as usize) as u32
    }

    pub fn num_planes(&self, plane_dim: u32) -> u32 {
        self.threads_per_block.div_ceil(plane_dim)
    }
}
