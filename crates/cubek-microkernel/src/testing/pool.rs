//! The max pool microkernel on its work item: an NHWC input `[1, h, w, c]` pooled into
//! `[1, oh, ow, c]`, every output cell the largest tap of its window, laid out as
//! [`WindowedLaunch`] says. The reduce reads the max's identity past the input's edge.

use crate::pool::PoolWorkItem;
use crate::pool::cube::{
    CubeMaxPoolMicrokernel, CubeMaxPoolMicrokernelSettings, cube_max_pool_microkernel,
};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

pub use super::window::{Channels, WindowAxis, WindowedLaunch};

/// A max pool microkernel ready to launch.
#[derive(Debug, Clone, Copy)]
pub struct PoolLaunch {
    pub microkernel: CubeMaxPoolMicrokernel,
    pub windowed: WindowedLaunch,
}

impl PoolLaunch {
    /// # Errors
    ///
    /// Why the device runs no max pool microkernel, or why the `channels` do not walk in whole
    /// blocks of the plane.
    pub fn new(
        hw: &Hardware,
        windows: [WindowAxis; 2],
        input: [usize; 2],
        channels: Channels,
    ) -> Result<Self, Refusal> {
        let windowed = WindowedLaunch::new(hw, windows, input, channels)?;
        let item = PoolWorkItem {
            channels: channels.per_unit,
            outputs: channels.per_unit,
        };
        let microkernel = CubeMaxPoolMicrokernel::best(hw, &item, &CubeMaxPoolMicrokernelSettings)?;
        Ok(Self {
            microkernel,
            windowed,
        })
    }

    /// `output[0, oh, ow, c]`, the largest of `input[0, oh · s + th · d - p, ow · s + tw · d - p,
    /// c]` over the taps inside the input, both NHWC and row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
        &self,
        client: &Client,
        input: TensorArg,
        output: TensorArg,
        element: ElemType,
    ) -> Result<(), Refusal> {
        let launcher = self.windowed.launcher(client)?;
        // SAFETY: both operands are bound through the launcher at the problem's own extents, the
        // input's reads past its edge masked, and the kernel indexes no buffer but through its
        // tiles.
        unsafe {
            cube_max_pool_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                self.windowed.gathered(input, element),
                self.windowed.output(output, element),
                launcher.partitioning_arg(),
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
fn cube_max_pool_kernel<E: Numeric>(
    input: &TileTensor<'_>,
    output: &TileTensor<'_>,
    partitioning: Partitioning,
    #[define(E)] _element: ElemType,
) {
    let input = input.tile(&partitioning);
    let output = output.tile(&partitioning);
    for cube in partitioning {
        for block in cube.walk() {
            cube_max_pool_microkernel(&block, &output.at(&block), &input.at(&block));
        }
    }
}
