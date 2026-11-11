//! The launch that fills a prompt's rings, which every conv ends with.

use super::kernel::ring_write_back_kernel;
use crate::kernels::conv::reads::{CausalConvReads, bound};
use crate::kernels::conv::{CausalConvPast, CausalConvPastInputs};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Space};
use cubek_microkernel::linear_recurrence::conv::cube::ChannelBox;
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, CHANNEL_BOX, CausalConvPartitioning, LINE_IN_BOX, RING_POSITION,
};
use cubek_std::routine::{CubeResources, Grid};

/// The write back of a prompt's rings, placed over the channels the calling conv convolves.
pub(crate) struct RingWriteBackLaunch {
    partitioning: CausalConvPartitioning,
    grid: Grid,
    width: usize,
    lines: ChannelBox,
}

impl RingWriteBackLaunch {
    /// The write back of `reads`' prompt on `hw`, its channels `width` values a line and its cubes
    /// of at most `units` units, or none for a window, which writes its own ring.
    ///
    /// # Errors
    ///
    /// What the device declines the launch's grid with.
    pub(crate) fn prepare(
        hw: &cubek_std::Hardware,
        reads: &CausalConvReads,
        width: usize,
        units: usize,
    ) -> Result<Option<Self>, Refusal> {
        if !matches!(reads.past(), CausalConvPast::Prompt { .. }) {
            return Ok(None);
        }
        let lines = reads.channels() / width;
        let units = units.min(lines);
        let boxes = lines.div_ceil(units);
        let mut axes = vec![
            (BATCH, reads.batch()),
            (RING_POSITION, reads.taps() - 1),
            (CHANNEL_BOX, boxes),
            (LINE_IN_BOX, units * width),
            (CHANNEL, reads.channels()),
        ];
        axes.extend(reads.past().past_axes());
        let leaf = [
            (RING_POSITION, 1),
            (CHANNEL_BOX, 1),
            (CHANNEL, reads.channels()),
            (LINE_IN_BOX, width),
        ];
        let partitioning = CausalConvPartitioning::new(
            Space::new(&axes),
            Levels::leaf(&leaf).units_distributed(LINE_IN_BOX, units),
            CHANNEL_BOX,
        );
        let cube = CubeResources {
            plane_width: units as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        Ok(Some(Self {
            partitioning,
            grid,
            width,
            lines: ChannelBox {
                units,
                channels: reads.channels(),
            },
        }))
    }

    /// Copies `reads`' prompt into its rings.
    pub(crate) fn launch(&self, reads: &CausalConvReads) {
        let CausalConvPastInputs::Prompt { valid, .. } = reads.past_inputs() else {
            unreachable!("only a prompt is prepared a write back");
        };
        let launcher = self.grid.launcher(reads.client(), || {
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
                self.grid.launch.cube_count.clone(),
                self.grid.launch.cube_dim,
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
