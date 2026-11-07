/// What one unit of an interpolation is handed: the output positions and channels it holds, each
/// reducing its taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateWorkItem {
    /// The channels of one output position a unit holds: its run along the contiguous axis.
    pub channels: usize,
    /// The output values a unit accumulates at once: its positions times its channels.
    pub outputs: usize,
}
