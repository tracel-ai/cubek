use crate::reduce::testing::{ComponentLaunch, LaunchedRows};
use crate::reduce::{
    IndexedReduceCarrierFamily, ReduceCarrier, ReduceCarrierFamily, ReduceWorkItem,
    TopKReduceCarrierFamily, TopValuesReduceCarrierFamily, ValuesReduceCarrierFamily,
};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tile::launch::{Decline, Grid};
use cubecl::tile::{Launcher, Levels, Partitioning, Space, TileTensorLaunch};
use cubek_std::Hardware;

/// Launches one reduce microkernel alone on rows.
pub(crate) trait MicrokernelLauncher: Copy + core::fmt::Debug {
    /// The launch its rows' owners are cut into.
    fn levels(&self, hw: &Hardware, rows: &LaunchedRows) -> Levels;

    /// The launch of its kernel into `F`'s carrier.
    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>);

    /// What every cell of its output holds before it runs.
    fn unwritten(&self) -> f32 {
        -7.0
    }

    /// The parts it splits a row into.
    fn parts(&self) -> usize {
        1
    }

    /// The units of a plane it reduces `item` with on `hw`.
    ///
    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    fn plane_width(&self, hw: &Hardware, _item: &ReduceWorkItem) -> Result<usize, Decline> {
        Ok(hw.plane_dim().unwrap_or(1) as usize)
    }

    /// What its launch counts as a plane: a cube's units where no level states one.
    fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
        rows.plane_lane_count
    }

    /// What its launch covers.
    fn space(&self, rows: &LaunchedRows) -> Space {
        rows.space()
    }

    /// The input bound over its dimensions, its parts among them where it splits a row.
    fn bound_input(&self, input: TensorArg, rows: &LaunchedRows) -> TileTensorLaunch {
        rows.bound_input(input, &rows.input_dimensions())
    }

    /// How far apart the values it reads of a row lie, on `hw`: every one unless it is one unit's
    /// share of them.
    fn stride_on(&self, _hw: &Hardware) -> usize {
        1
    }

    /// The values it reads of a row of `len`.
    fn values_of_value_count(&self, len: usize) -> usize {
        len
    }

    /// `out` written from `input`, both row-major, as [`LaunchedRows`] lays them.
    ///
    /// # Errors
    ///
    /// Why its levels do not launch on the device.
    fn launch(
        &self,
        client: &Client,
        input: TensorArg,
        out: TensorBinding,
        rows: LaunchedRows,
    ) -> Result<(), Decline> {
        let hw = Hardware::from_client(client);
        let space = self.space(&rows);
        let partitioning = Partitioning::new(space.clone(), self.levels(&hw, &rows).build());
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.launch_plane_width(&hw, &rows) as u32),
        };
        let launch = ComponentLaunch {
            client,
            launcher: Launcher::new(client, partitioning, &space, grid)?,
            input: self.bound_input(input, &rows),
            out,
            rows,
        };
        match rows.item.operation.carrier() {
            ReduceCarrier::Values => self.launch_into::<ValuesReduceCarrierFamily>(launch),
            ReduceCarrier::Indexed => self.launch_into::<IndexedReduceCarrierFamily>(launch),
            ReduceCarrier::TopK { .. } => self.launch_into::<TopKReduceCarrierFamily>(launch),
            ReduceCarrier::TopValues { .. } => {
                self.launch_into::<TopValuesReduceCarrierFamily>(launch)
            }
        }
        Ok(())
    }
}
