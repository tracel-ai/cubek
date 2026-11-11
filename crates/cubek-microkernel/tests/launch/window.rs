//! A window over a pooled or resampled axis, and the axes such a launch runs over: an NHWC input
//! gathered over every output position's taps.

use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::{
    Axis, Launcher, Levels, Partitioning, Projection, Space, TileSpec, TileTensorLaunch,
};
use cubek_std::Hardware;

pub(crate) const BATCH: Axis = Axis(0);
pub(crate) const OUTPUT_H: Axis = Axis(1);
pub(crate) const OUTPUT_W: Axis = Axis(2);
pub(crate) const TAP_H: Axis = Axis(3);
pub(crate) const TAP_W: Axis = Axis(4);
pub(crate) const CHANNEL: Axis = Axis(5);

/// How a window of one axis lands on the input: output position `o` and tap `t` read input
/// position `o · stride + t · dilation - padding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAxis {
    pub taps: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl WindowAxis {
    /// Output positions over an input `input` long.
    pub fn outputs(&self, input: usize) -> usize {
        (input + 2 * self.padding - self.dilation * (self.taps - 1) - 1) / self.stride + 1
    }

    /// The input axis this window reads, from `output` and `tap`.
    fn input_axis(&self, output: Axis, tap: Axis) -> PhysicalAxisMap {
        PhysicalAxisMap::affine(&[(output, self.stride), (tap, self.dilation)])
            .shifted(-(self.padding as isize))
    }
}

/// The channels of an input, and how many of them one unit takes at each step of the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channels {
    pub count: usize,
    pub per_unit: usize,
}

/// A launch over an NHWC input `[1, input[0], input[1], channels]` gathered under `windows`: a
/// plane's units take `channel_block` channels each, side by side, and every cube one output
/// position, walking its channels a block of the plane at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowedLaunch {
    /// The window along the rows, then the columns.
    pub windows: [WindowAxis; 2],
    /// The input's rows and columns.
    pub input: [usize; 2],
    pub channels: usize,
    /// Channels one unit takes at each step of the walk.
    pub channel_block: usize,
    pub plane_width: usize,
}

impl WindowedLaunch {
    /// # Errors
    ///
    /// Why the `channels` do not walk in whole blocks of the plane.
    pub(crate) fn new(
        hw: &Hardware,
        windows: [WindowAxis; 2],
        input: [usize; 2],
        channels: Channels,
    ) -> Result<Self, Refusal> {
        let Channels {
            count: channels,
            per_unit: channel_block,
        } = channels;
        let plane_width = hw.dispatch_plane_size() as usize;
        if !channels.is_multiple_of(plane_width * channel_block) {
            return Err(Refusal::Declined {
                why: "the channels walk in whole blocks of the plane",
            });
        }
        Ok(Self {
            windows,
            input,
            channels,
            channel_block,
            plane_width,
        })
    }

    /// The output's rows and columns.
    pub fn outputs(&self) -> [usize; 2] {
        [0, 1].map(|axis| self.windows[axis].outputs(self.input[axis]))
    }

    /// The launcher of the launch on `client`.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot place it.
    pub(crate) fn launcher(&self, client: &Client) -> Result<Launcher, Refusal> {
        let [rows, cols] = self.windows;
        let [out_h, out_w] = self.outputs();
        let space = Space::new(&[
            (BATCH, 1),
            (OUTPUT_H, out_h),
            (OUTPUT_W, out_w),
            (TAP_H, rows.taps),
            (TAP_W, cols.taps),
            (CHANNEL, self.channels),
        ]);
        let plane_units: Vec<_> = [(CHANNEL, self.plane_width)]
            .into_iter()
            .filter(|&(_, units)| units > 1)
            .collect();
        let partitioning = Partitioning::new(
            space.clone(),
            Levels::leaf(&[(OUTPUT_W, 1), (CHANNEL, self.channel_block), (OUTPUT_H, 1)])
                .units(&plane_units)
                .planes(&[(OUTPUT_H, 1)])
                .walk_every(&[CHANNEL])
                .cubes(&[OUTPUT_W, OUTPUT_H])
                .batches(&[BATCH])
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width as u32),
        };
        Launcher::new(client, partitioning, &space, grid)
    }

    /// `input` gathered over every output position's window, its reads past the edge masked.
    pub(crate) fn gathered(&self, input: TensorArg, element: ElemType) -> TileTensorLaunch {
        let [rows, cols] = self.windows;
        let projection = Projection::new(
            &[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL],
            &[
                PhysicalAxisMap::of(BATCH),
                rows.input_axis(OUTPUT_H, TAP_H),
                cols.input_axis(OUTPUT_W, TAP_W),
                PhysicalAxisMap::of(CHANNEL),
            ],
        );
        TileTensorLaunch::new(
            input,
            element,
            1,
            TileSpec::new(projection).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        )
    }

    /// `output` `[1, oh, ow, channels]`, row-major.
    pub(crate) fn output(&self, output: TensorArg, element: ElemType) -> TileTensorLaunch {
        TileTensorLaunch::new(
            output,
            element,
            1,
            TileSpec::direct(&[BATCH, OUTPUT_H, OUTPUT_W, CHANNEL]),
        )
    }
}
