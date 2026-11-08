/// What a pooling makes of a window: its largest tap, or the weighted sum of its taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolReduction {
    Max,
    Average,
}

/// What one unit of a pooling is handed: the output positions and channels it holds, each pooling
/// its window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolWorkItem {
    pub reduction: PoolReduction,
    /// The channels of one output position a unit holds: its run along the contiguous axis.
    pub channels: usize,
    /// The output values a unit holds at once: its positions times its channels.
    pub outputs: usize,
}
