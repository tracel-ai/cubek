//! The launch that fills a prompt's rings, which every conv ends with.

use super::kernel::ring_write_back_kernel;
use crate::kernels::conv::reads::{CausalConvReads, bound};
use crate::kernels::conv::{CausalConvPast, CausalConvPastInputs, ConvExtent};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Space};
use cubek_microkernel::linear_recurrence::conv::cube::ChannelBox;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, CHANNEL_BOX, CausalConvPartitioning, LINE_IN_BOX, RING_POSITION,
};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, CubeResources, Grid};

/// The write back of a prompt's rings, placed over the channels the calling conv convolves: what
/// the conv's blueprint holds of the launch that follows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWriteBack {
    pub partitioning: CausalConvPartitioning,
    /// Values a line of channels binds at.
    pub width: usize,
    pub lines: ChannelBox,
}

impl RingWriteBack {
    /// The write back of a prompt over `extent`, its channels `width` values a line, checked
    /// against `hw`, or none for a window, which writes its own ring.
    ///
    /// # Errors
    ///
    /// What the device declines the launch's grid with.
    pub(crate) fn prepare(
        hw: &Hardware,
        past: &CausalConvPast,
        extent: ConvExtent,
        width: usize,
    ) -> Result<Option<Self>, Refusal> {
        if !past.writes_ring_after() {
            return Ok(None);
        }
        let lines = extent.channels / width;
        let units = ChannelBox::most_units(hw).min(lines);
        let boxes = lines.div_ceil(units);
        let mut axes = vec![
            (BATCH, extent.batch),
            (RING_POSITION, extent.taps - 1),
            (CHANNEL_BOX, boxes),
            (LINE_IN_BOX, units * width),
            (CHANNEL, extent.channels),
        ];
        axes.extend(past.past_axes());
        let leaf = [
            (RING_POSITION, 1),
            (CHANNEL_BOX, 1),
            (CHANNEL, extent.channels),
            (LINE_IN_BOX, width),
        ];
        let partitioning = CausalConvPartitioning::new(
            Space::new(&axes),
            Levels::leaf(&leaf).units_distributed(LINE_IN_BOX, units),
            &[RING_POSITION, CHANNEL_BOX],
        );
        let write_back = Self {
            partitioning,
            width,
            lines: ChannelBox {
                units,
                channels: extent.channels,
            },
        };
        write_back.grid(hw)?;
        Ok(Some(write_back))
    }

    /// The cubes the launch spawns on `hw`.
    fn grid(&self, hw: &Hardware) -> Result<Grid, Refusal> {
        let cube = CubeResources {
            plane_width: self.lines.units as u32,
            shared_memory: 0,
        };
        Grid::new(hw, self.partitioning.as_cubek(), cube)
    }

    /// Copies `reads`' prompt into its rings, on the grid [`prepare`](Self::prepare) checked on
    /// `hw`.
    pub(crate) fn launch(&self, hw: &Hardware, reads: &CausalConvReads) {
        let grid = self.grid(hw).expect(ADMITTED_PLAN_RUNS);
        let CausalConvPastInputs::Prompt { valid, .. } = reads.past_inputs() else {
            unreachable!("only a prompt is prepared a write back");
        };
        let launcher = grid.launcher(reads.client(), || {
            (
                self.partitioning.kernel(),
                self.partitioning.as_cubek().space().clone(),
            )
        });
        let input = reads.bind_input(&launcher, self.width);
        let past = reads.past_inputs().bind(&launcher, self.width);
        let valid = bound(&launcher, valid, &[BATCH], 1);
        // SAFETY: every operand is bound through the launcher and indexed only through its tile,
        // whose reads past an edge are masked. A cube writes the ring columns of its own position
        // of its row and box, and the positions of a row take distinct columns, since the ring
        // holds the taps read back.
        unsafe {
            ring_write_back_kernel::launch_unchecked(
                reads.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                self.width,
                input,
                past.ring,
                past.lengths,
                valid,
                reads.past().ring().columns,
                launcher.partitioning_arg(),
                self.lines,
                reads.taps() - 1,
                reads.element(),
            );
        }
    }
}
