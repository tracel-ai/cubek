pub const RADIX_BITS: usize = 8;
pub const NUM_BUCKETS: usize = 1 << RADIX_BITS;
pub const DIGIT_MASK: u32 = (NUM_BUCKETS - 1) as u32;

#[derive(Clone, Debug)]
pub struct SortStrategy {
    pub items_per_thread: u32,
    pub threads_per_block: u32,
}

impl Default for SortStrategy {
    fn default() -> Self {
        Self {
            // Tuned for optimal performance on Metal/WGPU:
            // - 512 threads with 16 items/thread = 8192 items per block
            // - Larger blocks reduce scan kernel overhead (fewer blocks to process)
            // - Sweet spot between parallelism and overhead
            items_per_thread: 16,
            threads_per_block: 512,
        }
    }
}

impl SortStrategy {
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
