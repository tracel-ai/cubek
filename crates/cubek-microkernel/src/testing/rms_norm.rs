//! The RMS norm microkernels on their work item: `rows` rows of `dim`, each normalized by its own
//! root mean square and scaled by one weight row, `out = x / rms(x) · γ`. A plane takes a row, a
//! cube one row across its planes, a cube of groups one row a group; every output is written
//! straight into a plain output.

use crate::rms_norm::cube::{
    CubeGroupsRmsNormMicrokernel, CubeGroupsRmsNormMicrokernelSettings, CubeRmsNormMicrokernel,
    CubeRmsNormMicrokernelSettings, cube_groups_rms_norm_microkernel, cube_rms_norm_microkernel,
};
use crate::rms_norm::plane::{
    PlaneRmsNormMicrokernel, PlaneRmsNormMicrokernelSettings, plane_rms_norm_microkernel,
};
use crate::rms_norm::{B, D, R, RmsNormPartitioning, RmsNormScale, RmsNormWorkItem};
use crate::row_store::{CubeStage, OwnerLines, RowStore};
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// Which RMS norm microkernel normalizes the rows, at the settings it is picked at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormSettings {
    Plane(PlaneRmsNormMicrokernelSettings),
    Cube(CubeRmsNormMicrokernelSettings),
    CubeGroups(CubeGroupsRmsNormMicrokernelSettings),
}

/// The microkernel a device picks for [`RmsNormSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormMicrokernel {
    Plane(PlaneRmsNormMicrokernel),
    Cube(CubeRmsNormMicrokernel),
    CubeGroups(CubeGroupsRmsNormMicrokernel),
}

/// An RMS norm microkernel ready to launch over `rows` rows.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormLaunch {
    pub microkernel: RmsNormMicrokernel,
    pub item: RmsNormWorkItem,
    pub rows: usize,
}

impl RmsNormLaunch {
    /// The RMS norm microkernel `hw` runs at `settings` over `rows` rows of `dim` `element`s.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        element: ElemType,
        rows: usize,
        dim: usize,
        settings: &RmsNormSettings,
    ) -> Result<Self, Refusal> {
        let item = RmsNormWorkItem {
            element,
            dim,
            store: RowStore::Plain,
        };
        let microkernel = match settings {
            RmsNormSettings::Plane(settings) => {
                RmsNormMicrokernel::Plane(PlaneRmsNormMicrokernel::best(hw, &item, settings)?)
            }
            RmsNormSettings::Cube(settings) => {
                RmsNormMicrokernel::Cube(CubeRmsNormMicrokernel::best(hw, &item, settings)?)
            }
            RmsNormSettings::CubeGroups(settings) => RmsNormMicrokernel::CubeGroups(
                CubeGroupsRmsNormMicrokernel::best(hw, &item, settings)?,
            ),
        };
        Ok(Self {
            microkernel,
            item,
            rows,
        })
    }

    /// The width every operand binds at.
    pub fn width(&self) -> usize {
        match self.microkernel {
            RmsNormMicrokernel::Plane(microkernel) => microkernel.width,
            RmsNormMicrokernel::Cube(microkernel) => microkernel.plane.width,
            RmsNormMicrokernel::CubeGroups(microkernel) => microkernel.width,
        }
    }

    /// `out[r, d] = input[r, d] / sqrt(mean_d input[r, d]² + epsilon) · weight[d]`, `input` and
    /// `out` `[1, rows, dim]` and `weight` `[dim]`, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
        &self,
        client: &Client,
        input: TensorArg,
        weight: TensorArg,
        out: TensorArg,
        epsilon: f32,
    ) -> Result<(), Refusal> {
        let (store, dim) = (self.item.store, self.item.dim);
        let space = Space::new(&[(B, 1), (R, self.rows), (D, dim)]);
        let (partitioning, plane_width) = match self.microkernel {
            RmsNormMicrokernel::Plane(microkernel) => (
                RmsNormPartitioning::over_planes(
                    space.clone(),
                    Levels::leaf(&[(R, 1), (D, dim)]),
                    PlaneOrder::new(&[(R, 1)], CubeOrder::RowMajor),
                ),
                microkernel.lanes,
            ),
            RmsNormMicrokernel::Cube(microkernel) => (
                RmsNormPartitioning::new(
                    space.clone(),
                    Levels::leaf(&[(R, 1), (D, microkernel.plane.run)])
                        .planes(&[(store.cut(), microkernel.planes)]),
                    store,
                ),
                microkernel.plane.lanes,
            ),
            RmsNormMicrokernel::CubeGroups(microkernel) => (
                RmsNormPartitioning::new(
                    space.clone(),
                    Levels::leaf(&[(R, 1), (D, dim)]).planes(&[(R, 1)]),
                    store,
                ),
                microkernel.units,
            ),
        };
        let partitioning = partitioning.as_tile().clone();
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(plane_width as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let (element, width) = (self.item.element, self.width());
        let bound =
            |arg, axes: &[_]| TileTensorLaunch::new(arg, element, width, TileSpec::direct(axes));
        let (input, weight, out) = (
            bound(input, &[B, R, D]),
            bound(weight, &[D]),
            bound(out, &[B, R, D]),
        );
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; a plain store stages nothing.
        unsafe {
            match self.microkernel {
                RmsNormMicrokernel::Plane(microkernel) => plane_rms_norm_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    width,
                    input,
                    weight,
                    out,
                    epsilon,
                    launcher.partitioning_arg(),
                    microkernel,
                    element,
                ),
                RmsNormMicrokernel::Cube(microkernel) => cube_rms_norm_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    width,
                    input,
                    weight,
                    out,
                    epsilon,
                    launcher.partitioning_arg(),
                    microkernel,
                    element,
                ),
                RmsNormMicrokernel::CubeGroups(microkernel) => {
                    cube_groups_rms_norm_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        width,
                        input,
                        weight,
                        out,
                        epsilon,
                        launcher.partitioning_arg(),
                        microkernel,
                        element,
                    )
                }
            }
        }
        Ok(())
    }
}

/// The one plane normalizing its row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn plane_rms_norm_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let (x, w, o) = (
        input.tile(&partitioning),
        weight.tile(&partitioning),
        out.tile(&partitioning),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for plane in &partitioning {
        let mut row_out = OwnerLines::window(o.at(&plane));
        plane_rms_norm_microkernel::<E, N>(
            &x.at(&plane),
            &w.at(&plane),
            &mut row_out,
            &scale,
            microkernel,
        );
    }
}

/// Every cube normalizing its row across its planes, written straight into the output.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cube_rms_norm_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let (x, w, o) = (
        input.tile(&partitioning),
        weight.tile(&partitioning),
        out.tile(&partitioning),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.plane.dim);
    for cube in &partitioning {
        cube_rms_norm_microkernel::<E, N>(
            &cube,
            &x.at(&cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            CubeStage::Direct,
        );
    }
}

/// Every cube normalizing its group's row, written straight into the output.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cube_groups_rms_norm_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeGroupsRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let (x, w, o) = (
        input.tile(&partitioning),
        weight.tile(&partitioning),
        out.tile(&partitioning),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for cube in &partitioning {
        cube_groups_rms_norm_microkernel::<E, N>(
            &cube,
            &x.at(&cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            1usize,
            CubeStage::Direct,
        );
    }
}
