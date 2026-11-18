//! The RMS norm microkernels on their work item: `rows` rows of `dim`, each normalized by its own
//! root mean square and scaled by one weight row, `out = x / rms(x) · γ`. A plane takes a row, a
//! cube one row across its planes, a cube of groups one row a group; every output is written
//! straight into a plain output. A gate multiplies by `silu(gate)` the row before the norm or its
//! output after it, and the output may be written at another element than the input's.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::rms_norm::cube::{
    CubeGroupsRmsNormMicrokernel, CubeGroupsRmsNormMicrokernelSettings, CubeRmsNormMicrokernel,
    CubeRmsNormMicrokernelSettings, cube_groups_rms_norm_microkernel, cube_rms_norm_microkernel,
};
use cubek_microkernel::rms_norm::plane::{
    PlaneRmsNormMicrokernel, PlaneRmsNormMicrokernelSettings, plane_rms_norm_microkernel,
};
use cubek_microkernel::rms_norm::{
    B, D, Gate, GateOrder, GatedRow, R, RmsNormPartitioning, RmsNormScale, RmsNormWorkItem,
};
use cubek_microkernel::row_store::{CubeStage, OwnerLines, RowStore};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::RowBatch;

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

/// The element a norm reads its input, weight and gate at, and the one it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormElements {
    pub input: ElemType,
    pub output: ElemType,
}

/// An RMS norm microkernel ready to launch over `rows` rows.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormLaunch {
    pub microkernel: RmsNormMicrokernel,
    pub item: RmsNormWorkItem,
    pub rows: usize,
}

impl RmsNormLaunch {
    /// The RMS norm microkernel `hw` runs at `settings` over `batch`'s rows, read and written at
    /// `elements` and gated as `gate` says.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: RmsNormElements,
        gate: Option<GateOrder>,
        batch: RowBatch,
        settings: &RmsNormSettings,
    ) -> Result<Self, Refusal> {
        let RowBatch { rows, dim } = batch;
        let item = RmsNormWorkItem {
            element: elements.input,
            output_element: elements.output,
            dim,
            store: RowStore::Plain,
            gate,
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

    /// `out[r, d] = input[r, d] / sqrt(mean_d input[r, d]² + epsilon) · weight[d]`, `input`,
    /// `gate` where the norm is gated, and `out` `[1, rows, dim]` and `weight` `[dim]`, all
    /// row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
        &self,
        client: &Client,
        input: TensorArg,
        weight: TensorArg,
        gate: Option<TensorArg>,
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
        let (held, written) = (self.item.held_element(), self.item.output_element);
        let bound = |arg, element, axes: &[_]| {
            TileTensorLaunch::new(arg, element, width, TileSpec::direct(axes))
        };
        let (input, weight, out) = (
            bound(input, element, &[B, R, D]),
            bound(weight, element, &[D]),
            bound(out, written, &[B, R, D]),
        );
        let gate = gate.map(|gate| bound(gate, element, &[B, R, D]));
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
                    gate.into(),
                    out,
                    epsilon,
                    launcher.partitioning_arg(),
                    microkernel,
                    element,
                    held,
                    written,
                ),
                RmsNormMicrokernel::Cube(microkernel) => cube_rms_norm_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    width,
                    input,
                    weight,
                    gate.into(),
                    out,
                    epsilon,
                    launcher.partitioning_arg(),
                    microkernel,
                    element,
                    held,
                    written,
                ),
                RmsNormMicrokernel::CubeGroups(microkernel) => {
                    cube_groups_rms_norm_kernel::launch_unchecked(
                        client,
                        launcher.cube_count(),
                        launcher.cube_dim(),
                        width,
                        input,
                        weight,
                        gate.into(),
                        out,
                        epsilon,
                        launcher.partitioning_arg(),
                        microkernel,
                        element,
                        held,
                        written,
                    )
                }
            }
        }
        Ok(())
    }
}

/// The one plane normalizing its row.
#[cube(launch_unchecked)]
fn plane_rms_norm_kernel<E: Float, Held: Float, Out: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(Held)] _held_element: ElemType,
    #[define(Out)] _output_element: ElemType,
) {
    let (w, o) = (weight.tile(&partitioning), out.tile(&partitioning));
    let gated = GatedRow::new(
        &input.tile(&partitioning),
        Gate::bound(gate.map(|gate| gate.tile(&partitioning)), microkernel.gate),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for plane in &partitioning {
        let mut row_out = OwnerLines::window(o.at(&plane));
        plane_rms_norm_microkernel::<E, Held, Out, N>(
            &gated.at(&plane),
            &w.at(&plane),
            &mut row_out,
            &scale,
            microkernel,
        );
    }
}

/// Every cube normalizing its row across its planes, written straight into the output.
#[cube(launch_unchecked)]
fn cube_rms_norm_kernel<E: Float, Held: Float, Out: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(Held)] _held_element: ElemType,
    #[define(Out)] _output_element: ElemType,
) {
    let (w, o) = (weight.tile(&partitioning), out.tile(&partitioning));
    let gated = GatedRow::new(
        &input.tile(&partitioning),
        Gate::bound(
            gate.map(|gate| gate.tile(&partitioning)),
            microkernel.plane.gate,
        ),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.plane.dim);
    for cube in &partitioning {
        cube_rms_norm_microkernel::<E, Held, Out, N>(
            &cube,
            &gated.at(&cube),
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
fn cube_groups_rms_norm_kernel<E: Float, Held: Float, Out: Float, N: Size>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    gate: ComptimeOption<&TileTensor<'static>>,
    out: &TileTensor<'_>,
    epsilon: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeGroupsRmsNormMicrokernel,
    #[define(E)] _element: ElemType,
    #[define(Held)] _held_element: ElemType,
    #[define(Out)] _output_element: ElemType,
) {
    let (w, o) = (weight.tile(&partitioning), out.tile(&partitioning));
    let gated = GatedRow::new(
        &input.tile(&partitioning),
        Gate::bound(gate.map(|gate| gate.tile(&partitioning)), microkernel.gate),
    );
    let scale = RmsNormScale::new(epsilon, microkernel.dim);
    for cube in &partitioning {
        cube_groups_rms_norm_microkernel::<E, Held, Out, N>(
            &cube,
            &gated.at(&cube),
            &w.at(&cube),
            &o.at(&cube),
            &scale,
            microkernel,
            1usize,
            CubeStage::Direct,
        );
    }
}
