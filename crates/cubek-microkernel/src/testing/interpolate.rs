//! The interpolate microkernel on its work item: an NHWC input `[1, h, w, c]` resampled into
//! `[1, oh, ow, c]`, every output cell its window's taps weighted by `weights[0, oh, ow, th, tw]`
//! and summed, laid out as [`WindowedLaunch`] says. A tap past the input's edge reads zero.
//!
//! The weights are a tensor here, not a filter's: what the microkernel computes is the
//! contraction, whatever weights it is handed.

use crate::interpolate::InterpolateWorkItem;
use crate::interpolate::cube::{
    CubeInterpolateMicrokernel, CubeInterpolateMicrokernelSettings, cube_interpolate_microkernel,
};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Partitioning, TileSpec, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::window::{BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W};
pub use super::window::{Channels, WindowAxis, WindowedLaunch};

/// An interpolate microkernel ready to launch.
#[derive(Debug, Clone, Copy)]
pub struct InterpolateLaunch {
    pub microkernel: CubeInterpolateMicrokernel,
    pub windowed: WindowedLaunch,
}

impl InterpolateLaunch {
    /// # Errors
    ///
    /// Why the device runs no interpolate microkernel, or why the `channels` do not walk in whole
    /// blocks of the plane.
    pub fn new(
        hw: &Hardware,
        windows: [WindowAxis; 2],
        input: [usize; 2],
        channels: Channels,
    ) -> Result<Self, Refusal> {
        let windowed = WindowedLaunch::new(hw, windows, input, channels)?;
        let item = InterpolateWorkItem {
            channels: channels.per_unit,
            outputs: channels.per_unit,
        };
        let microkernel =
            CubeInterpolateMicrokernel::best(hw, &item, &CubeInterpolateMicrokernelSettings)?;
        Ok(Self {
            microkernel,
            windowed,
        })
    }

    /// `output[0, oh, ow, c] = Σ weights[0, oh, ow, th, tw] · input[0, oh · s + th · d - p,
    /// ow · s + tw · d - p, c]`, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
        &self,
        client: &Client,
        input: TensorArg,
        weights: TensorArg,
        output: TensorArg,
        element: ElemType,
    ) -> Result<(), Refusal> {
        let launcher = self.windowed.launcher(client)?;
        let weights = TileTensorLaunch::new(
            weights,
            element,
            1,
            TileSpec::direct(&[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W]),
        );
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // input's reads past its edge masked, and the kernel indexes no buffer but through its
        // tiles.
        unsafe {
            cube_interpolate_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                self.windowed.gathered(input, element),
                weights,
                self.windowed.output(output, element),
                launcher.partitioning_arg(),
                self.microkernel,
                element,
            );
        }
        Ok(())
    }
}

/// Every cube walking its channels, a block at a time.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
fn cube_interpolate_kernel<E: Float>(
    input: &TileTensor<'_>,
    weights: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeInterpolateMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let weights = weights.tile(&partitioning);
    let output = output.tile(&partitioning);
    for cube in partitioning {
        for block in cube.walk() {
            cube_interpolate_microkernel(
                &block,
                &output.at(&block),
                &weights.at(&block),
                &input.at(&block),
                microkernel,
            );
        }
    }
}
