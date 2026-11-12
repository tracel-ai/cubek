//! The launch that fills a prompt's rings, which the conv ends with.

use super::kernel::ring_write_back_kernel;
use crate::kernels::bound::bound;
use crate::kernels::conv::{CausalConvPastInputs, CausalConvProblem, CausalConvTensors};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Space};
use cubek_microkernel::linear_recurrence::conv::cube::{ChannelBox, RingAdvance};
use cubek_microkernel::linear_recurrence::{
    BATCH, CHANNEL, CHANNEL_BOX, CausalConvPartitioning, LINE_IN_BOX, RING_POSITION,
};
use cubek_std::Hardware;
use cubek_std::routine::{CubeResources, Grid};
use std::sync::Arc;

/// The write back of a prompt's rings, placed over the channels the conv convolves: what the
/// conv's blueprint holds of the launch that follows it. It keeps the grid and launcher its
/// preparation settled, so a prepared conv fills rings without placing anything again.
#[derive(Debug, Clone)]
pub struct RingWriteBackLaunch {
    partitioning: CausalConvPartitioning,
    width: usize,
    channel_box: ChannelBox,
    grid: Grid,
}

impl PartialEq for RingWriteBackLaunch {
    /// The placement: the grid follows from it on a device.
    fn eq(&self, other: &Self) -> bool {
        self.partitioning == other.partitioning
            && self.width == other.width
            && self.channel_box == other.channel_box
    }
}

impl Eq for RingWriteBackLaunch {}

impl RingWriteBackLaunch {
    /// The write back of a prompt of `problem`, its channels `width` values a line over the
    /// conv's `channel_box`, checked against `hw`, or none for a window, which writes its own ring.
    ///
    /// # Errors
    ///
    /// What the device declines the launch's grid with.
    pub(crate) fn prepare(
        hw: &Hardware,
        problem: &CausalConvProblem,
        width: usize,
        channel_box: ChannelBox,
    ) -> Result<Option<Arc<Self>>, Refusal> {
        if problem.past.advance() == RingAdvance::InLaunch {
            return Ok(None);
        }
        let channels = problem.channels;
        let units = channel_box.units;
        let boxes = (channels / width).div_ceil(units);
        let mut axes = vec![
            (BATCH, problem.batch),
            (RING_POSITION, problem.taps - 1),
            (CHANNEL_BOX, boxes),
            (LINE_IN_BOX, units * width),
            (CHANNEL, channels),
        ];
        axes.extend(problem.past.past_axes());
        let leaf = [
            (RING_POSITION, 1),
            (CHANNEL_BOX, 1),
            (CHANNEL, channels),
            (LINE_IN_BOX, width),
        ];
        let partitioning = CausalConvPartitioning::new(
            Space::new(&axes),
            Levels::leaf(&leaf).units_distributed(LINE_IN_BOX, units),
            &[RING_POSITION, CHANNEL_BOX],
        );
        let cube = CubeResources {
            plane_width: units as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        Ok(Some(Arc::new(Self {
            partitioning,
            width,
            channel_box,
            grid,
        })))
    }

    /// Copies `tensors`' prompt into its rings.
    pub(crate) fn launch(&self, tensors: &CausalConvTensors) {
        let grid = &self.grid;
        let CausalConvPastInputs::Prompt { valid, .. } = tensors.past_inputs() else {
            unreachable!("only a prompt is prepared a write back");
        };
        let problem = tensors.problem();
        let launcher = self.partitioning.launcher(tensors.client(), grid);
        let input = tensors.bind_input(&launcher, self.width);
        let past = tensors.bind_ring(&launcher, self.width);
        let valid = bound(&launcher, valid, &[BATCH], 1);
        // SAFETY: every operand is bound through the launcher. The input is read only at
        // positions below `valid`, which the kernel clamps to the prompt's positions. A cube writes
        // the ring columns of its own position of its row and box, and the positions of a row take
        // distinct columns, since the ring holds the taps read back.
        unsafe {
            ring_write_back_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count.clone(),
                grid.launch.cube_dim,
                self.width,
                input,
                past.store,
                past.lengths,
                valid,
                problem.past.ring().columns,
                problem.past.positions(),
                launcher.partitioning_arg(),
                self.channel_box,
                problem.taps - 1,
                problem.element,
            );
        }
    }
}
