/// What one unit of a max pooling is handed: the output positions and channels it holds, each
/// taking the largest tap of its window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolWorkItem {
    /// The channels of one output position a unit holds: its run along the contiguous axis.
    pub channels: usize,
    /// The output values a unit holds at once: its positions times its channels.
    pub outputs: usize,
}
