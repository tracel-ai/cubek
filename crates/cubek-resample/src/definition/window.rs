use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub struct WindowArgs {
    /// Number of taps.
    pub size: usize,
    /// Tap spacing.
    pub dilation: usize,
}

impl WindowArgs {
    pub fn new(size: usize) -> WindowArgs {
        WindowArgs { size, dilation: 0 }
    }

    pub fn with_dilation(mut self, dilation: usize) -> WindowArgs {
        self.dilation = dilation;
        self
    }

    pub fn to_launch<R: Runtime>(&self) -> WindowArgsLaunch<R> {
        WindowArgsLaunch::new(self.size, self.dilation)
    }
}
