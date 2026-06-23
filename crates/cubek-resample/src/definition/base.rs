use crate::definition::{
    BoundaryMode, Kernel, NormalizationMode, Placement, PlacementArgs, Semiring, WindowArgs,
};
use cubecl::prelude::*;

/// Resampling args.
#[derive(CubeType, CubeLaunch)]
pub struct ResampleArgs {
    pub resample_axes: Sequence<ResampleAxisArgs>,
}

impl Default for ResampleArgs {
    fn default() -> Self {
        Self {
            resample_axes: Sequence::new(),
        }
    }
}

impl ResampleArgs {
    pub fn with_resample_axis_args(mut self, resample_axis_args: ResampleAxisArgs) -> Self {
        self.resample_axes.push(resample_axis_args);
        self
    }

    pub fn to_launch<R: Runtime>(self) -> ResampleArgsLaunch<R> {
        let mut resample_axes_launch = SequenceArg::new();
        for resample_axes_args in self.resample_axes.iter() {
            resample_axes_launch.push(resample_axes_args.to_launch::<R>());
        }
        ResampleArgsLaunch::new(resample_axes_launch)
    }
}

/// Resampling axis args.
#[derive(CubeType, CubeLaunch)]
pub struct ResampleAxisArgs {
    pub window_args: WindowArgs,
    pub placement_args: PlacementArgs,
}

impl ResampleAxisArgs {
    pub fn new(window_args: WindowArgs, placement_args: PlacementArgs) -> Self {
        Self {
            window_args,
            placement_args,
        }
    }

    pub fn to_launch<R: Runtime>(&self) -> ResampleAxisArgsLaunch<R> {
        ResampleAxisArgsLaunch::new(
            self.window_args.to_launch::<R>(),
            self.placement_args.to_launch::<R>(),
        )
    }
}

/// Resampling operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub struct Resample {
    pub resample_axes: Sequence<ResampleAxis>,
    pub semiring: Semiring,
    pub boundary: BoundaryMode,
    pub normalization: NormalizationMode,
}

impl Resample {
    pub fn new(
        semiring: Semiring,
        boundary: BoundaryMode,
        normalization: NormalizationMode,
    ) -> Self {
        Self {
            resample_axes: Sequence::new(),
            semiring,
            boundary,
            normalization,
        }
    }

    /// Order matters, last axis added is innermost.
    pub fn with_axis(mut self, axis: ResampleAxis) -> Self {
        self.resample_axes.push(axis);
        self
    }

    /// Returns the number of lanes to unroll: `vector_size` if the vectorized axis
    /// is a resampling axis, otherwise `1`.
    pub fn compute_num_lanes(&self, vectorized_axis: usize, vector_size: usize) -> usize {
        let mut is_vectorized = false;

        for axis_idx in 0..self.num_axes() {
            is_vectorized |= self.resample_axes[axis_idx].axis == vectorized_axis;
        }

        if is_vectorized { vector_size } else { 1_usize }
    }

    pub fn num_axes(&self) -> usize {
        self.resample_axes.len()
    }
}

#[cube]
impl Resample {
    /// Calculates the total number of taps.
    pub fn calculate_num_taps(args: &ResampleArgs, #[comptime] config: &Resample) -> usize {
        let mut num_taps = 1;

        #[unroll]
        for axis_idx in 0..config.num_axes() {
            num_taps *= args.resample_axes.index(axis_idx).window_args.size
        }

        num_taps
    }
}

/// Resample axis operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub struct ResampleAxis {
    pub axis: usize,
    pub kernel: Kernel,
    pub placement: Placement,
}

impl ResampleAxis {
    pub fn new(axis: usize, kernel: Kernel, placement: Placement) -> Self {
        Self {
            axis,
            kernel,
            placement,
        }
    }
}
